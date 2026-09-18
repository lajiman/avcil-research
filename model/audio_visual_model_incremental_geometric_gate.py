"""
AV-CIL incremental audio-visual model with an explicit prototype-geometric
calibration gate for audio-guided visual attention.

The original AV-CIL attention is first evaluated as a full-strength proposal:
    v_guided = audio-guided visual z1 feature
and compared against the audio-independent endpoint:
    v_uniform = uniform-pooled visual z1 feature

A detached class-conditional geometric contribution is computed in a shared
visual prototype space.  The resulting suppress-only gate calibrates the final
visual z1 feature used by z2/classification:
    v_final = g * v_guided + (1 - g) * v_uniform

The original guided visual feature is still exposed for AV-CIL's z1
contrastive loss, and the original attention maps are still exposed for
attention-score distillation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LSCLinear, SplitLSCLinear


class IncreAudioVisualNet(nn.Module):
    def __init__(self, args, step_out_class_num: int, LSC: bool = False):
        super().__init__()

        self.args = args
        self.modality = args.modality
        self.num_classes = int(step_out_class_num)

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

            # Optional projection heads retained from the user's current model.
            self.use_z1_cm_projection_head = getattr(args, "z1_cm_projection_head", False)
            if self.use_z1_cm_projection_head:
                proj_dim = getattr(args, "z1_cm_projection_dim", 768)
                hidden_dim = getattr(args, "z1_cm_projection_hidden_dim", 768)
                head_type = getattr(args, "z1_cm_projection_type", "mlp")
                self.z1_audio_projector = self._build_z1_projection_head(
                    in_dim=768,
                    hidden_dim=hidden_dim,
                    out_dim=proj_dim,
                    head_type=head_type,
                )
                self.z1_visual_projector = self._build_z1_projection_head(
                    in_dim=768,
                    hidden_dim=hidden_dim,
                    out_dim=proj_dim,
                    head_type=head_type,
                )

            # -------------------------------------------------------------
            # Explicit geometric gate configuration.
            # -------------------------------------------------------------
            self.use_geometric_gate = bool(getattr(args, "geometric_gate", False))
            self.geo_prototype_temperature = float(
                getattr(args, "geo_prototype_temperature", 0.1)
            )
            self.geo_posterior_temperature = float(
                getattr(args, "geo_posterior_temperature", 1.0)
            )
            self.geo_gate_temperature = float(
                getattr(args, "geo_gate_temperature", 1.0)
            )
            self.geo_gate_topk = int(getattr(args, "geo_gate_topk", 3))
            self.geo_gate_min = float(getattr(args, "geo_gate_min", 0.0))

            if self.geo_prototype_temperature <= 0:
                raise ValueError("geo_prototype_temperature must be > 0")
            if self.geo_posterior_temperature <= 0:
                raise ValueError("geo_posterior_temperature must be > 0")
            if self.geo_gate_temperature <= 0:
                raise ValueError("geo_gate_temperature must be > 0")
            if self.geo_gate_topk <= 0:
                raise ValueError("geo_gate_topk must be > 0")
            if not 0.0 <= self.geo_gate_min < 1.0:
                raise ValueError("geo_gate_min must be in [0, 1)")

            # The training script deliberately controls when the gate is active.
            # It is disabled for task 0 and during the warm-up of later tasks.
            self.geo_gate_enabled = False

            # Detached class-level reference geometry.  Buffers are saved in the
            # whole-model checkpoint and expanded together with the classifier.
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

    # ------------------------------------------------------------------
    # Generic helpers retained from the user's current model.
    # ------------------------------------------------------------------
    def _build_z1_projection_head(
        self,
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

        audio_z = F.normalize(self.z1_audio_projector(audio_feature), dim=1)
        visual_z = F.normalize(self.z1_visual_projector(visual_feature), dim=1)
        return audio_z, visual_z

    # ------------------------------------------------------------------
    # Prototype/gate state management.
    # ------------------------------------------------------------------
    def set_geometric_gate_enabled(self, enabled: bool) -> None:
        if self.modality != "audio-visual":
            return
        self.geo_gate_enabled = bool(enabled) and bool(self.use_geometric_gate)

    def set_geometric_prototypes(
        self,
        prototypes: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        counts: Optional[torch.Tensor] = None,
    ) -> None:
        """Install a detached visual prototype bank.

        Args:
            prototypes: shape=(num_seen_classes, 768).
            valid_mask: optional bool mask, shape=(num_seen_classes,).
            counts: optional per-class sample counts.
        """
        if self.modality != "audio-visual":
            raise ValueError("Geometric prototypes are only defined for audio-visual mode")
        if prototypes.ndim != 2 or prototypes.shape[1] != 768:
            raise ValueError(
                "prototypes must have shape (num_seen_classes, 768), got {}".format(
                    tuple(prototypes.shape)
                )
            )
        if prototypes.shape[0] != self.num_classes:
            raise ValueError(
                "prototype count {} does not match model.num_classes {}".format(
                    prototypes.shape[0], self.num_classes
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

        # Invalid rows remain exactly zero, preventing accidental use.
        proto = torch.where(valid.unsqueeze(1), proto, torch.zeros_like(proto))
        self.geo_visual_prototypes.copy_(proto)
        self.geo_prototype_valid.copy_(valid)

        if counts is None:
            count_tensor = valid.to(dtype=torch.float32)
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
        """Compute CMCDR-style target-vs-rest B for every candidate class.

        scores: shape=(B, C), where C >= 2.
        returns: shape=(B, C), B_c = s_c - logsumexp_{k != c}(s_k).
        """
        if scores.ndim != 2:
            raise ValueError("scores must be a 2-D tensor")
        num_classes = scores.shape[1]
        if num_classes < 2:
            return torch.zeros_like(scores)

        expanded = scores.unsqueeze(1).expand(-1, num_classes, -1)
        diagonal_mask = torch.eye(
            num_classes,
            dtype=torch.bool,
            device=scores.device,
        ).unsqueeze(0)
        other_scores = expanded.masked_fill(diagonal_mask, float("-inf"))
        logsumexp_others = torch.logsumexp(other_scores, dim=-1)
        return scores - logsumexp_others

    def _compute_geometric_gate(
        self,
        audio_feature: torch.Tensor,
        visual_guided_feature: torch.Tensor,
        visual_uniform_feature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute a detached proposal gate from the two z1 visual endpoints."""
        batch_size = audio_feature.shape[0]
        num_classes = self.num_classes
        device = audio_feature.device
        dtype = audio_feature.dtype

        ones = torch.ones(batch_size, device=device, dtype=dtype)
        zeros = torch.zeros(batch_size, device=device, dtype=dtype)
        zero_matrix = torch.zeros(batch_size, num_classes, device=device, dtype=dtype)

        valid = self.geo_prototype_valid[:num_classes]
        if (
            not self.use_geometric_gate
            or valid.numel() != num_classes
            or int(valid.sum().item()) < 2
        ):
            return {
                "proposed_gate": ones,
                "geo_score": zeros,
                "geo_contribution": zero_matrix,
                "neutral_posterior": zero_matrix,
            }

        # The entire routing computation is deliberately detached.  The model can
        # train the guided/uniform endpoints through the final mixture, but cannot
        # manipulate its own gate through d(g)/d(R).
        with torch.no_grad():
            prototypes = F.normalize(
                self.geo_visual_prototypes[:num_classes],
                dim=1,
                eps=1e-12,
            )
            guided = F.normalize(visual_guided_feature.detach(), dim=1, eps=1e-12)
            uniform = F.normalize(visual_uniform_feature.detach(), dim=1, eps=1e-12)

            score_guided = torch.matmul(guided, prototypes.t())
            score_uniform = torch.matmul(uniform, prototypes.t())
            score_guided = score_guided / self.geo_prototype_temperature
            score_uniform = score_uniform / self.geo_prototype_temperature

            # Invalid candidates cannot participate in either B or posterior routing.
            invalid = ~valid
            score_guided[:, invalid] = -1.0e4
            score_uniform[:, invalid] = -1.0e4

            benefit_guided = self._benefit_matrix_from_scores(score_guided)
            benefit_uniform = self._benefit_matrix_from_scores(score_uniform)
            contribution = benefit_guided - benefit_uniform
            contribution[:, invalid] = 0.0

            # Neutral class posterior: both modalities are present, but audio-guided
            # visual attention has not yet been used in this branch.
            neutral_logits = self.classifier(
                audio_feature.detach() + visual_uniform_feature.detach()
            )
            neutral_logits = neutral_logits / self.geo_posterior_temperature
            neutral_logits[:, invalid] = -1.0e4
            posterior = F.softmax(neutral_logits, dim=1)

            topk = min(self.geo_gate_topk, int(valid.sum().item()))
            top_prob, top_idx = torch.topk(posterior, k=topk, dim=1)
            top_prob = top_prob / top_prob.sum(dim=1, keepdim=True).clamp_min(1e-12)
            selected_contribution = contribution.gather(1, top_idx)
            sample_score = torch.sum(top_prob * selected_contribution, dim=1)

            # Suppress-only calibration: positive/non-harmful evidence preserves the
            # original AVCIL guided branch (g=1); negative evidence moves toward the
            # uniform branch.  Clamp avoids numerical underflow.
            negative_score = torch.minimum(sample_score, torch.zeros_like(sample_score))
            exponent = torch.clamp(
                negative_score / self.geo_gate_temperature,
                min=-20.0,
                max=0.0,
            )
            proposed_gate = torch.exp(exponent)
            if self.geo_gate_min > 0.0:
                proposed_gate = self.geo_gate_min + (
                    1.0 - self.geo_gate_min
                ) * proposed_gate

        return {
            "proposed_gate": proposed_gate.to(dtype=dtype),
            "geo_score": sample_score.to(dtype=dtype),
            "geo_contribution": contribution.to(dtype=dtype),
            "neutral_posterior": posterior.to(dtype=dtype),
        }

    # ------------------------------------------------------------------
    # Forward paths.
    # ------------------------------------------------------------------
    def forward(
        self,
        visual=None,
        audio=None,
        out_logits: bool = True,
        out_features: bool = False,
        out_features_norm: bool = False,
        out_feature_before_fusion: bool = False,
        out_z1_projection: bool = False,
        out_attn_score: bool = False,
        AFC_train_out: bool = False,
        return_dict: bool = False,
        out_analysis_features: bool = False,
        out_gate_details: bool = False,
        skip_geometric_gate: bool = False,
    ):
        if self.modality == "visual":
            if visual is None:
                raise ValueError("input frames are None when modality contains visual")
            visual_feature = F.relu(self.visual_proj(torch.mean(visual, dim=1)))
            logits = self.classifier(visual_feature)

            if return_dict:
                result: Dict[str, torch.Tensor] = {}
                if AFC_train_out:
                    visual_feature.retain_grad()
                    result["logits"] = logits
                    result["visual_feature"] = visual_feature
                    return result
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
                outputs += (
                    F.normalize(visual_feature, dim=1)
                    if out_features_norm
                    else visual_feature,
                )
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
                    result["logits"] = logits
                    result["audio_feature"] = audio_feature
                    return result
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
                outputs += (
                    F.normalize(audio_feature, dim=1)
                    if out_features_norm
                    else audio_feature,
                )
            return outputs[0] if len(outputs) == 1 else outputs

        # ------------------------ audio-visual ------------------------
        if visual is None:
            raise ValueError("input frames are None when modality contains visual")
        if audio is None:
            raise ValueError("input audio are None when modality contains audio")

        visual_4d = visual.reshape(visual.shape[0], 8, -1, 768)

        # Two gate-independent visual endpoints.
        visual_uniform_pooled = visual_4d.mean(dim=(1, 2))
        spatial_attn_score, temporal_attn_score = self.audio_visual_attention(
            audio,
            visual_4d,
        )
        visual_guided_pooled = torch.sum(spatial_attn_score * visual_4d, dim=2)
        visual_guided_pooled = torch.sum(
            temporal_attn_score * visual_guided_pooled,
            dim=1,
        )

        audio_feature = F.relu(self.audio_proj(audio))
        visual_guided_feature = F.relu(self.visual_proj(visual_guided_pooled))
        visual_uniform_feature = F.relu(self.visual_proj(visual_uniform_pooled))

        if skip_geometric_gate:
            batch_size = audio_feature.shape[0]
            gate_info = {
                "proposed_gate": torch.ones(
                    batch_size, device=audio_feature.device, dtype=audio_feature.dtype
                ),
                "geo_score": torch.zeros(
                    batch_size, device=audio_feature.device, dtype=audio_feature.dtype
                ),
                "geo_contribution": torch.zeros(
                    batch_size, self.num_classes,
                    device=audio_feature.device, dtype=audio_feature.dtype
                ),
                "neutral_posterior": torch.zeros(
                    batch_size, self.num_classes,
                    device=audio_feature.device, dtype=audio_feature.dtype
                ),
            }
        else:
            gate_info = self._compute_geometric_gate(
                audio_feature=audio_feature,
                visual_guided_feature=visual_guided_feature,
                visual_uniform_feature=visual_uniform_feature,
            )
        proposed_gate = gate_info["proposed_gate"]
        if self.geo_gate_enabled and self.use_geometric_gate:
            effective_gate = proposed_gate
        else:
            effective_gate = torch.ones_like(proposed_gate)

        visual_final_feature = (
            effective_gate.unsqueeze(1) * visual_guided_feature
            + (1.0 - effective_gate).unsqueeze(1) * visual_uniform_feature
        )

        guided_fusion = audio_feature + visual_guided_feature
        uniform_fusion = audio_feature + visual_uniform_feature
        final_fusion = audio_feature + visual_final_feature

        logits = self.classifier(final_fusion)

        # These two endpoints are useful for diagnostics.  The uniform logits are
        # also the neutral branch used by the gate, but are recomputed with normal
        # precision and a clear output contract here.
        guided_logits = self.classifier(guided_fusion)
        uniform_logits = self.classifier(uniform_fusion)

        if return_dict:
            result = {}

            if AFC_train_out:
                audio_feature.retain_grad()
                visual_guided_feature.retain_grad()
                visual_guided_pooled.retain_grad()
                result["logits"] = logits
                result["visual_pooled_feature"] = visual_guided_pooled
                result["audio_feature"] = audio_feature
                result["visual_feature"] = visual_guided_feature
                return result

            if out_logits:
                result["logits"] = logits

            if out_features:
                result["features"] = (
                    F.normalize(final_fusion, dim=1)
                    if out_features_norm
                    else final_fusion
                )

            if out_feature_before_fusion:
                # Preserve baseline training semantics: z1 contrastive learning sees
                # the original full-strength guided visual proposal, not v_final.
                result["audio_feature"] = F.normalize(audio_feature, dim=1)
                result["visual_feature"] = F.normalize(
                    visual_guided_feature,
                    dim=1,
                )

            if out_z1_projection:
                if not getattr(self, "use_z1_cm_projection_head", False):
                    raise ValueError(
                        "out_z1_projection=True but z1_cm_projection_head is not enabled"
                    )
                audio_z1_proj, visual_z1_proj = self.project_z1_features(
                    audio_feature,
                    visual_guided_feature,
                )
                result["audio_z1_proj"] = audio_z1_proj
                result["visual_z1_proj"] = visual_z1_proj

            if out_attn_score:
                result["spatial_attn_score"] = spatial_attn_score
                result["temporal_attn_score"] = temporal_attn_score

            if out_gate_details:
                result["geo_gate_enabled"] = torch.tensor(
                    float(self.geo_gate_enabled and self.use_geometric_gate),
                    device=logits.device,
                )
                result["geo_gate"] = effective_gate
                result["geo_gate_proposed"] = proposed_gate
                result["geo_score"] = gate_info["geo_score"]
                result["geo_contribution"] = gate_info["geo_contribution"]
                result["neutral_posterior"] = gate_info["neutral_posterior"]
                result["guided_logits"] = guided_logits
                result["uniform_logits"] = uniform_logits

            if out_analysis_features:
                result["z0_audio_raw"] = audio
                result["z0_visual_uniform_raw"] = visual_uniform_pooled
                result["z0_audio_norm"] = F.normalize(audio, dim=1)
                result["z0_visual_uniform_norm"] = F.normalize(
                    visual_uniform_pooled,
                    dim=1,
                )

                result["attn_visual_pooled_raw"] = visual_guided_pooled
                result["attn_visual_pooled_norm"] = F.normalize(
                    visual_guided_pooled,
                    dim=1,
                )

                # Backward-compatible names keep z1_visual_raw as guided.
                result["z1_audio_raw"] = audio_feature
                result["z1_visual_raw"] = visual_guided_feature
                result["z1_visual_guided_raw"] = visual_guided_feature
                result["z1_visual_uniform_raw"] = visual_uniform_feature
                result["z1_visual_final_raw"] = visual_final_feature
                result["z2_fusion_raw"] = final_fusion
                result["z2_fusion_guided_raw"] = guided_fusion
                result["z2_fusion_uniform_raw"] = uniform_fusion

                result["z1_audio_norm"] = F.normalize(audio_feature, dim=1)
                result["z1_visual_norm"] = F.normalize(
                    visual_guided_feature,
                    dim=1,
                )
                result["z1_visual_guided_norm"] = F.normalize(
                    visual_guided_feature,
                    dim=1,
                )
                result["z1_visual_uniform_norm"] = F.normalize(
                    visual_uniform_feature,
                    dim=1,
                )
                result["z1_visual_final_norm"] = F.normalize(
                    visual_final_feature,
                    dim=1,
                )
                result["z2_fusion_norm"] = F.normalize(final_fusion, dim=1)

            return result

        # Original tuple/tensor interface for compatibility with existing utilities.
        outputs = ()
        if AFC_train_out:
            audio_feature.retain_grad()
            visual_guided_feature.retain_grad()
            visual_guided_pooled.retain_grad()
            return logits, visual_guided_pooled, audio_feature, visual_guided_feature

        if out_logits:
            outputs += (logits,)
        if out_features:
            outputs += (
                F.normalize(final_fusion, dim=1)
                if out_features_norm
                else final_fusion,
            )
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
                audio_feature,
                visual_guided_feature,
            )
        if out_attn_score:
            outputs += (spatial_attn_score, temporal_attn_score)

        return outputs[0] if len(outputs) == 1 else outputs

    def audio_visual_attention(
        self,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        proj_audio_features = torch.tanh(self.attn_audio_proj(audio_features))
        proj_visual_features = torch.tanh(self.attn_visual_proj(visual_features))

        spatial_score = torch.einsum(
            "ijkd,id->ijkd",
            [proj_visual_features, proj_audio_features],
        )
        spatial_attn_score = F.softmax(spatial_score, dim=2)
        spatial_attned_proj_visual_features = torch.sum(
            spatial_attn_score * proj_visual_features,
            dim=2,
        )

        temporal_score = torch.einsum(
            "ijd,id->ijd",
            [spatial_attned_proj_visual_features, proj_audio_features],
        )
        temporal_attn_score = F.softmax(temporal_score, dim=1)
        return spatial_attn_score, temporal_attn_score

    def incremental_classifier(self, numclass: int) -> None:
        """Expand the classifier and the prototype buffers together."""
        numclass = int(numclass)
        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        in_features = self.classifier.in_features
        out_features = self.classifier.out_features
        device = weight.device
        dtype = weight.dtype

        self.classifier = nn.Linear(in_features, numclass, bias=True).to(
            device=device,
            dtype=dtype,
        )
        self.classifier.weight.data[:out_features].copy_(weight)
        self.classifier.bias.data[:out_features].copy_(bias)

        if self.modality == "audio-visual":
            old_proto = self.geo_visual_prototypes
            old_valid = self.geo_prototype_valid
            old_counts = self.geo_prototype_counts

            new_proto = torch.zeros(
                numclass,
                768,
                device=old_proto.device,
                dtype=old_proto.dtype,
            )
            new_valid = torch.zeros(
                numclass,
                device=old_valid.device,
                dtype=torch.bool,
            )
            new_counts = torch.zeros(
                numclass,
                device=old_counts.device,
                dtype=old_counts.dtype,
            )

            keep = min(out_features, numclass)
            new_proto[:keep].copy_(old_proto[:keep])
            new_valid[:keep].copy_(old_valid[:keep])
            new_counts[:keep].copy_(old_counts[:keep])

            self.geo_visual_prototypes = new_proto
            self.geo_prototype_valid = new_valid
            self.geo_prototype_counts = new_counts

        self.num_classes = numclass

    def extract_joint_feature(self, visual=None, audio=None) -> torch.Tensor:
        """Return the model's current fused feature, shape=(B, 768)."""
        if self.modality == "visual":
            if visual is None:
                raise ValueError("input frames are None when modality contains visual")
            return F.relu(self.visual_proj(torch.mean(visual, dim=1)))

        if self.modality == "audio":
            if audio is None:
                raise ValueError("input audio are None when modality contains audio")
            return F.relu(self.audio_proj(audio))

        result = self.forward(
            visual=visual,
            audio=audio,
            out_logits=False,
            out_features=True,
            return_dict=True,
        )
        return result["features"]

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)