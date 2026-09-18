"""
AV-CIL model with class-conditional geometric guidance contribution and an
exact linear logit-routing implementation.

For each sample i and candidate class c:

    R_geo[i, c] = B_geo(v_guided)[i, c] - B_geo(v_uniform)[i, c]
    g[i, c]     = exp(min(R_geo[i, c], 0) / T_gate)

The original AV-CIL classifier is affine and fusion is additive. Therefore the
class-c logit can be decomposed exactly as:

    l*[i, c] = l_audio[i, c]
              + g[i, c] * l_visual_guided[i, c]
              + (1-g[i, c]) * l_visual_uniform[i, c]

where the classifier bias is included once in l_audio.  No ground-truth class,
pseudo-label, posterior, or top-k routing is required at test time.

The geometric gate is detached: it is recomputed dynamically on every forward,
but gradients do not flow through R_geo -> g.  Classification/KD gradients do
flow through the routed guided and uniform branches.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LSCLinear, SplitLSCLinear  # kept for repository compatibility


class IncreAudioVisualNet(nn.Module):
    def __init__(self, args, step_out_class_num: int, LSC: bool = False):
        super().__init__()

        self.args = args
        self.modality = args.modality
        self.num_classes = int(step_out_class_num)
        self._lsc_requested = bool(LSC)

        if self.modality not in {"visual", "audio", "audio-visual"}:
            raise ValueError("modality must be 'visual', 'audio' or 'audio-visual'")

        if self.modality == "visual":
            self.visual_proj = nn.Linear(768, 768)

        elif self.modality == "audio":
            self.audio_proj = nn.Linear(768, 768)

        else:
            self.audio_proj = nn.Linear(768, 768)
            self.visual_proj = nn.Linear(768, 768)
            self.attn_audio_proj = nn.Linear(768, 768)
            self.attn_visual_proj = nn.Linear(768, 768)

            # Optional z1 projection heads retained from the supplied model.
            self.use_z1_cm_projection_head = bool(
                getattr(args, "z1_cm_projection_head", False)
            )
            if self.use_z1_cm_projection_head:
                proj_dim = int(getattr(args, "z1_cm_projection_dim", 768))
                hidden_dim = int(
                    getattr(args, "z1_cm_projection_hidden_dim", 768)
                )
                head_type = str(getattr(args, "z1_cm_projection_type", "mlp"))
                self.z1_audio_projector = self._build_z1_projection_head(
                    768, hidden_dim, proj_dim, head_type
                )
                self.z1_visual_projector = self._build_z1_projection_head(
                    768, hidden_dim, proj_dim, head_type
                )

            # Geometric linear routing configuration.
            self.use_geometric_linear_routing = bool(
                getattr(args, "geometric_linear_routing", False)
            )
            self.geo_prototype_temperature = float(
                getattr(args, "geo_prototype_temperature", 0.1)
            )
            self.geo_gate_temperature = float(
                getattr(args, "geo_gate_temperature", 1.0)
            )
            self.geo_gate_min = float(getattr(args, "geo_gate_min", 0.0))

            if self.geo_prototype_temperature <= 0:
                raise ValueError("geo_prototype_temperature must be > 0")
            if self.geo_gate_temperature <= 0:
                raise ValueError("geo_gate_temperature must be > 0")
            if not 0.0 <= self.geo_gate_min < 1.0:
                raise ValueError("geo_gate_min must be in [0, 1)")

            # Controlled by the training script. Task 0 and warm-up use baseline.
            self.geo_routing_enabled = False

            # Detached uniform-visual reference prototypes.
            self.register_buffer(
                "geo_visual_prototypes",
                torch.zeros(self.num_classes, 768, dtype=torch.float32),
            )
            self.register_buffer(
                "geo_prototype_valid",
                torch.zeros(self.num_classes, dtype=torch.bool),
            )
            self.register_buffer(
                "geo_prototype_counts",
                torch.zeros(self.num_classes, dtype=torch.float32),
            )

        if LSC:
            self.classifier = LSCLinear(768, self.num_classes)
        else:
            self.classifier = nn.Linear(768, self.num_classes)

        if (
            self.modality == "audio-visual"
            and self.use_geometric_linear_routing
            and not isinstance(self.classifier, nn.Linear)
        ):
            raise TypeError(
                "geometric_linear_routing requires nn.Linear classifier; "
                "LSC/nonlinear heads are not supported by this exact routine"
            )

    # ------------------------------------------------------------------
    # Optional z1 projection heads
    # ------------------------------------------------------------------
    @staticmethod
    def _build_z1_projection_head(
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        head_type: str = "mlp",
    ) -> nn.Module:
        if head_type == "linear":
            return nn.Linear(in_dim, out_dim)
        if head_type == "mlp":
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )
        raise ValueError("z1_cm_projection_type must be 'linear' or 'mlp'")

    def project_z1_features(
        self,
        audio_feature: torch.Tensor,
        visual_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not getattr(self, "use_z1_cm_projection_head", False):
            raise ValueError("z1_cm_projection_head is not enabled in this model")
        return (
            F.normalize(self.z1_audio_projector(audio_feature), dim=1),
            F.normalize(self.z1_visual_projector(visual_feature), dim=1),
        )

    # ------------------------------------------------------------------
    # Routing/prototype state
    # ------------------------------------------------------------------
    def set_geometric_routing_enabled(self, enabled: bool) -> None:
        if self.modality != "audio-visual":
            return
        self.geo_routing_enabled = bool(enabled) and bool(
            self.use_geometric_linear_routing
        )

    # Backward-friendly alias for utilities written for the prior gate version.
    def set_geometric_gate_enabled(self, enabled: bool) -> None:
        self.set_geometric_routing_enabled(enabled)

    def set_geometric_prototypes(
        self,
        prototypes: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        counts: Optional[torch.Tensor] = None,
    ) -> None:
        if self.modality != "audio-visual":
            raise ValueError("Geometric prototypes require audio-visual mode")
        if prototypes.shape != (self.num_classes, 768):
            raise ValueError(
                "prototypes must have shape ({}, 768), got {}".format(
                    self.num_classes, tuple(prototypes.shape)
                )
            )

        proto = prototypes.detach().to(
            device=self.geo_visual_prototypes.device,
            dtype=self.geo_visual_prototypes.dtype,
        )
        proto = F.normalize(proto, dim=1, eps=1e-12)

        if valid_mask is None:
            valid = torch.isfinite(proto).all(dim=1) & (proto.norm(dim=1) > 0)
        else:
            valid = valid_mask.detach().to(
                device=self.geo_prototype_valid.device,
                dtype=torch.bool,
            )
            if valid.shape != (self.num_classes,):
                raise ValueError("valid_mask has an incompatible shape")

        proto = torch.where(valid.unsqueeze(1), proto, torch.zeros_like(proto))
        self.geo_visual_prototypes.copy_(proto)
        self.geo_prototype_valid.copy_(valid)

        if counts is None:
            count_tensor = valid.float()
        else:
            count_tensor = counts.detach().to(
                device=self.geo_prototype_counts.device,
                dtype=torch.float32,
            )
            if count_tensor.shape != (self.num_classes,):
                raise ValueError("counts has an incompatible shape")
        self.geo_prototype_counts.copy_(count_tensor)

    @staticmethod
    def _benefit_matrix_from_scores(scores: torch.Tensor) -> torch.Tensor:
        """Return B_c = s_c - logsumexp_{k != c}(s_k) for all c in O(BC)."""
        if scores.ndim != 2:
            raise ValueError("scores must have shape (batch, classes)")
        if scores.shape[1] < 2:
            return torch.zeros_like(scores)

        # B_c = log p_c - log(1-p_c), with p=softmax(scores).
        log_p = F.log_softmax(scores, dim=1)
        p = log_p.exp().clamp(min=0.0, max=1.0 - 1e-7)
        return log_p - torch.log1p(-p)

    def _compute_classwise_geometric_gate(
        self,
        visual_guided_feature: torch.Tensor,
        visual_uniform_feature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute detached sample-and-class gate matrix, shape=(B, C)."""
        batch_size = visual_guided_feature.shape[0]
        device = visual_guided_feature.device
        dtype = visual_guided_feature.dtype
        num_classes = self.num_classes

        ones = torch.ones(batch_size, num_classes, device=device, dtype=dtype)
        zeros = torch.zeros_like(ones)
        valid = self.geo_prototype_valid[:num_classes]

        if (
            not self.use_geometric_linear_routing
            or valid.numel() != num_classes
            or int(valid.sum().item()) < 2
        ):
            return {
                "proposed_gate": ones,
                "geo_contribution": zeros,
                "geo_b_guided": zeros,
                "geo_b_uniform": zeros,
            }

        # R -> g is intentionally detached. The gate changes on every forward as
        # features change, but the model cannot manipulate g through d(g)/d(R).
        with torch.no_grad():
            prototypes = F.normalize(
                self.geo_visual_prototypes[:num_classes].float(),
                dim=1,
                eps=1e-12,
            )
            guided = F.normalize(
                visual_guided_feature.detach().float(), dim=1, eps=1e-12
            )
            uniform = F.normalize(
                visual_uniform_feature.detach().float(), dim=1, eps=1e-12
            )

            score_guided = guided @ prototypes.t()
            score_uniform = uniform @ prototypes.t()
            score_guided = score_guided / self.geo_prototype_temperature
            score_uniform = score_uniform / self.geo_prototype_temperature

            invalid = ~valid
            score_guided[:, invalid] = -1.0e4
            score_uniform[:, invalid] = -1.0e4

            b_guided = self._benefit_matrix_from_scores(score_guided)
            b_uniform = self._benefit_matrix_from_scores(score_uniform)
            contribution = b_guided - b_uniform
            contribution[:, invalid] = 0.0

            negative = torch.minimum(contribution, torch.zeros_like(contribution))
            exponent = torch.clamp(
                negative / self.geo_gate_temperature,
                min=-20.0,
                max=0.0,
            )
            proposed_gate = torch.exp(exponent)
            if self.geo_gate_min > 0.0:
                proposed_gate = self.geo_gate_min + (
                    1.0 - self.geo_gate_min
                ) * proposed_gate
            proposed_gate[:, invalid] = 1.0

        return {
            "proposed_gate": proposed_gate.to(dtype=dtype),
            "geo_contribution": contribution.to(dtype=dtype),
            "geo_b_guided": b_guided.to(dtype=dtype),
            "geo_b_uniform": b_uniform.to(dtype=dtype),
        }

    def _linear_logit_components(
        self,
        audio_feature: torch.Tensor,
        visual_guided_feature: torch.Tensor,
        visual_uniform_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(self.classifier, nn.Linear):
            raise TypeError(
                "Exact geometric linear routing requires self.classifier to be nn.Linear"
            )
        weight = self.classifier.weight
        bias = self.classifier.bias
        audio_logits = F.linear(audio_feature, weight, bias)
        guided_visual_logits = F.linear(visual_guided_feature, weight, None)
        uniform_visual_logits = F.linear(visual_uniform_feature, weight, None)
        return audio_logits, guided_visual_logits, uniform_visual_logits

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        visual=None,
        audio=None,
        out_logits=True,
        out_features=False,
        out_features_norm=False,
        out_feature_before_fusion=False,
        out_z1_projection=False,
        out_attn_score=False,
        AFC_train_out=False,
        return_dict=False,
        out_analysis_features=False,
        out_gate_details=False,
        skip_geometric_routing=False,
        skip_geometric_gate=False,  # alias used by earlier utilities
    ):
        if self.modality == "visual":
            if visual is None:
                raise ValueError("input frames are None when modality contains visual")
            visual_feature = F.relu(self.visual_proj(torch.mean(visual, dim=1)))
            logits = self.classifier(visual_feature)
            if return_dict:
                result = {}
                if AFC_train_out:
                    visual_feature.retain_grad()
                    return {"logits": logits, "visual_feature": visual_feature}
                if out_logits:
                    result["logits"] = logits
                if out_features:
                    result["features"] = (
                        F.normalize(visual_feature, dim=1)
                        if out_features_norm
                        else visual_feature
                    )
                return result
            outputs = ()
            if AFC_train_out:
                visual_feature.retain_grad()
                return logits, visual_feature
            if out_logits:
                outputs += (logits,)
            if out_features:
                outputs += ((
                    F.normalize(visual_feature, dim=1)
                    if out_features_norm
                    else visual_feature
                ),)
            return outputs[0] if len(outputs) == 1 else outputs

        if self.modality == "audio":
            if audio is None:
                raise ValueError("input audio are None when modality contains audio")
            audio_feature = F.relu(self.audio_proj(audio))
            logits = self.classifier(audio_feature)
            if return_dict:
                result = {}
                if AFC_train_out:
                    audio_feature.retain_grad()
                    return {"logits": logits, "audio_feature": audio_feature}
                if out_logits:
                    result["logits"] = logits
                if out_features:
                    result["features"] = (
                        F.normalize(audio_feature, dim=1)
                        if out_features_norm
                        else audio_feature
                    )
                return result
            outputs = ()
            if AFC_train_out:
                audio_feature.retain_grad()
                return logits, audio_feature
            if out_logits:
                outputs += (logits,)
            if out_features:
                outputs += ((
                    F.normalize(audio_feature, dim=1)
                    if out_features_norm
                    else audio_feature
                ),)
            return outputs[0] if len(outputs) == 1 else outputs

        if visual is None:
            raise ValueError("input frames are None when modality contains visual")
        if audio is None:
            raise ValueError("input audio are None when modality contains audio")

        visual_4d = visual.reshape(visual.shape[0], 8, -1, 768)
        visual_uniform_pooled = visual_4d.mean(dim=(1, 2))

        spatial_attn_score, temporal_attn_score = self.audio_visual_attention(
            audio, visual_4d
        )
        visual_guided_pooled = torch.sum(spatial_attn_score * visual_4d, dim=2)
        visual_guided_pooled = torch.sum(
            temporal_attn_score * visual_guided_pooled, dim=1
        )

        audio_feature = F.relu(self.audio_proj(audio))
        visual_guided_feature = F.relu(self.visual_proj(visual_guided_pooled))
        visual_uniform_feature = F.relu(self.visual_proj(visual_uniform_pooled))

        skip_routing = bool(skip_geometric_routing or skip_geometric_gate)
        if skip_routing:
            batch_size = audio_feature.shape[0]
            routing_info = {
                "proposed_gate": torch.ones(
                    batch_size,
                    self.num_classes,
                    device=audio_feature.device,
                    dtype=audio_feature.dtype,
                ),
                "geo_contribution": torch.zeros(
                    batch_size,
                    self.num_classes,
                    device=audio_feature.device,
                    dtype=audio_feature.dtype,
                ),
                "geo_b_guided": torch.zeros(
                    batch_size,
                    self.num_classes,
                    device=audio_feature.device,
                    dtype=audio_feature.dtype,
                ),
                "geo_b_uniform": torch.zeros(
                    batch_size,
                    self.num_classes,
                    device=audio_feature.device,
                    dtype=audio_feature.dtype,
                ),
            }
        else:
            routing_info = self._compute_classwise_geometric_gate(
                visual_guided_feature=visual_guided_feature,
                visual_uniform_feature=visual_uniform_feature,
            )

        proposed_gate = routing_info["proposed_gate"]
        if self.geo_routing_enabled and self.use_geometric_linear_routing:
            effective_gate = proposed_gate
        else:
            effective_gate = torch.ones_like(proposed_gate)

        guided_fusion = audio_feature + visual_guided_feature
        uniform_fusion = audio_feature + visual_uniform_feature

        # Direct baseline endpoints, useful for validation/diagnostics.
        guided_logits = self.classifier(guided_fusion)
        uniform_logits = self.classifier(uniform_fusion)

        if self.geo_routing_enabled and self.use_geometric_linear_routing:
            audio_logits, guided_visual_logits, uniform_visual_logits = (
                self._linear_logit_components(
                    audio_feature,
                    visual_guided_feature,
                    visual_uniform_feature,
                )
            )
            logits = (
                audio_logits
                + effective_gate * guided_visual_logits
                + (1.0 - effective_gate) * uniform_visual_logits
            )
        else:
            # Exact original AVCIL forward during task 0 and warm-up.
            audio_logits = self.classifier(audio_feature)
            guided_visual_logits = F.linear(
                visual_guided_feature,
                self.classifier.weight,
                None,
            ) if isinstance(self.classifier, nn.Linear) else torch.zeros_like(guided_logits)
            uniform_visual_logits = F.linear(
                visual_uniform_feature,
                self.classifier.weight,
                None,
            ) if isinstance(self.classifier, nn.Linear) else torch.zeros_like(uniform_logits)
            logits = guided_logits

        if return_dict:
            result: Dict[str, torch.Tensor] = {}

            if AFC_train_out:
                audio_feature.retain_grad()
                visual_guided_feature.retain_grad()
                visual_guided_pooled.retain_grad()
                return {
                    "logits": logits,
                    "visual_pooled_feature": visual_guided_pooled,
                    "audio_feature": audio_feature,
                    "visual_feature": visual_guided_feature,
                }

            if out_logits:
                result["logits"] = logits

            if out_features:
                # Class-wise routing does not correspond to one shared z2 feature.
                # Guided fusion is returned only for backward-compatible analysis.
                result["features"] = (
                    F.normalize(guided_fusion, dim=1)
                    if out_features_norm
                    else guided_fusion
                )

            if out_feature_before_fusion:
                # Keep original AVCIL contrastive semantics.
                result["audio_feature"] = F.normalize(audio_feature, dim=1)
                result["visual_feature"] = F.normalize(
                    visual_guided_feature, dim=1
                )

            if out_z1_projection:
                if not getattr(self, "use_z1_cm_projection_head", False):
                    raise ValueError(
                        "out_z1_projection=True but z1_cm_projection_head is not enabled"
                    )
                audio_z, visual_z = self.project_z1_features(
                    audio_feature, visual_guided_feature
                )
                result["audio_z1_proj"] = audio_z
                result["visual_z1_proj"] = visual_z

            if out_attn_score:
                result["spatial_attn_score"] = spatial_attn_score
                result["temporal_attn_score"] = temporal_attn_score

            if out_gate_details:
                result["geo_routing_enabled"] = torch.tensor(
                    float(
                        self.geo_routing_enabled
                        and self.use_geometric_linear_routing
                    ),
                    device=logits.device,
                )
                result["geo_gate"] = effective_gate
                result["geo_gate_proposed"] = proposed_gate
                result["geo_contribution"] = routing_info["geo_contribution"]
                result["geo_b_guided"] = routing_info["geo_b_guided"]
                result["geo_b_uniform"] = routing_info["geo_b_uniform"]
                result["guided_logits"] = guided_logits
                result["uniform_logits"] = uniform_logits
                result["audio_logits"] = audio_logits
                result["guided_visual_logits"] = guided_visual_logits
                result["uniform_visual_logits"] = uniform_visual_logits

            if out_analysis_features:
                result["z0_audio_raw"] = audio
                result["z0_visual_uniform_raw"] = visual_uniform_pooled
                result["z0_audio_norm"] = F.normalize(audio, dim=1)
                result["z0_visual_uniform_norm"] = F.normalize(
                    visual_uniform_pooled, dim=1
                )
                result["attn_visual_pooled_raw"] = visual_guided_pooled
                result["attn_visual_pooled_norm"] = F.normalize(
                    visual_guided_pooled, dim=1
                )
                result["z1_audio_raw"] = audio_feature
                result["z1_visual_raw"] = visual_guided_feature
                result["z1_visual_guided_raw"] = visual_guided_feature
                result["z1_visual_uniform_raw"] = visual_uniform_feature
                result["z2_fusion_raw"] = guided_fusion
                result["z2_fusion_guided_raw"] = guided_fusion
                result["z2_fusion_uniform_raw"] = uniform_fusion
                result["z1_audio_norm"] = F.normalize(audio_feature, dim=1)
                result["z1_visual_norm"] = F.normalize(
                    visual_guided_feature, dim=1
                )
                result["z1_visual_guided_norm"] = F.normalize(
                    visual_guided_feature, dim=1
                )
                result["z1_visual_uniform_norm"] = F.normalize(
                    visual_uniform_feature, dim=1
                )
                result["z2_fusion_norm"] = F.normalize(guided_fusion, dim=1)

            return result

        outputs = ()
        if AFC_train_out:
            audio_feature.retain_grad()
            visual_guided_feature.retain_grad()
            visual_guided_pooled.retain_grad()
            return logits, visual_guided_pooled, audio_feature, visual_guided_feature

        if out_logits:
            outputs += (logits,)
        if out_features:
            outputs += ((
                F.normalize(guided_fusion, dim=1)
                if out_features_norm
                else guided_fusion
            ),)
        if out_feature_before_fusion:
            outputs += (
                F.normalize(audio_feature, dim=1),
                F.normalize(visual_guided_feature, dim=1),
            )
        if out_z1_projection:
            if not getattr(self, "use_z1_cm_projection_head", False):
                raise ValueError(
                    "out_z1_projection=True but z1_cm_projection_head is not enabled"
                )
            outputs += self.project_z1_features(
                audio_feature, visual_guided_feature
            )
        if out_attn_score:
            outputs += (spatial_attn_score, temporal_attn_score)
        return outputs[0] if len(outputs) == 1 else outputs

    # ------------------------------------------------------------------
    # Attention and feature helpers
    # ------------------------------------------------------------------
    def audio_visual_attention(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        proj_audio_features = torch.tanh(self.attn_audio_proj(audio_features))
        proj_visual_features = torch.tanh(self.attn_visual_proj(visual_features))

        spatial_score = torch.einsum(
            "ijkd,id->ijkd", [proj_visual_features, proj_audio_features]
        )
        spatial_attn_score = F.softmax(spatial_score, dim=2)
        spatial_attned = torch.sum(
            spatial_attn_score * proj_visual_features, dim=2
        )
        temporal_score = torch.einsum(
            "ijd,id->ijd", [spatial_attned, proj_audio_features]
        )
        temporal_attn_score = F.softmax(temporal_score, dim=1)
        return spatial_attn_score, temporal_attn_score

    def extract_uniform_visual_feature(self, visual: torch.Tensor) -> torch.Tensor:
        """Efficient z1 uniform-visual feature extractor for prototype refresh."""
        if visual is None:
            raise ValueError("visual input is required")
        visual_4d = visual.reshape(visual.shape[0], 8, -1, 768)
        pooled = visual_4d.mean(dim=(1, 2))
        return F.relu(self.visual_proj(pooled))

    def incremental_classifier(self, numclass: int) -> None:
        if not isinstance(self.classifier, nn.Linear):
            raise TypeError(
                "This experiment's incremental classifier expects nn.Linear"
            )
        numclass = int(numclass)
        old_weight = self.classifier.weight.data
        old_bias = self.classifier.bias.data
        old_out = self.classifier.out_features
        in_features = self.classifier.in_features
        device = old_weight.device
        dtype = old_weight.dtype

        new_classifier = nn.Linear(in_features, numclass, bias=True).to(
            device=device, dtype=dtype
        )
        keep = min(old_out, numclass)
        new_classifier.weight.data[:keep].copy_(old_weight[:keep])
        new_classifier.bias.data[:keep].copy_(old_bias[:keep])
        self.classifier = new_classifier

        if self.modality == "audio-visual":
            old_proto = self.geo_visual_prototypes
            old_valid = self.geo_prototype_valid
            old_counts = self.geo_prototype_counts
            new_proto = torch.zeros(
                numclass, 768, device=old_proto.device, dtype=old_proto.dtype
            )
            new_valid = torch.zeros(
                numclass, device=old_valid.device, dtype=torch.bool
            )
            new_counts = torch.zeros(
                numclass, device=old_counts.device, dtype=old_counts.dtype
            )
            new_proto[:keep].copy_(old_proto[:keep])
            new_valid[:keep].copy_(old_valid[:keep])
            new_counts[:keep].copy_(old_counts[:keep])
            self.geo_visual_prototypes = new_proto
            self.geo_prototype_valid = new_valid
            self.geo_prototype_counts = new_counts

        self.num_classes = numclass

    def extract_joint_feature(self, visual=None, audio=None) -> torch.Tensor:
        """Return baseline guided fused feature for stage-2 compatibility."""
        if self.modality == "visual":
            if visual is None:
                raise ValueError("visual input is required")
            return F.relu(self.visual_proj(torch.mean(visual, dim=1)))
        if self.modality == "audio":
            if audio is None:
                raise ValueError("audio input is required")
            return F.relu(self.audio_proj(audio))
        if visual is None or audio is None:
            raise ValueError("visual and audio inputs are required")

        visual_4d = visual.reshape(visual.shape[0], 8, -1, 768)
        spatial, temporal = self.audio_visual_attention(audio, visual_4d)
        pooled = torch.sum(spatial * visual_4d, dim=2)
        pooled = torch.sum(temporal * pooled, dim=1)
        audio_feature = F.relu(self.audio_proj(audio))
        visual_feature = F.relu(self.visual_proj(pooled))
        return audio_feature + visual_feature

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)