from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_visual_model_incremental import IncreAudioVisualNet as BaseIncreAudioVisualNet


class IncreAudioVisualNet(BaseIncreAudioVisualNet):
    """Current AVCIL model plus a label-conditioned O1 oracle class gate.

    The class deliberately subclasses the user's current
    ``audio_visual_model_incremental.IncreAudioVisualNet`` so that all current
    projection heads, analysis helpers and classifier utilities remain
    available.

    For a sample with ground-truth label ``y``:

        v_gate = g_y * v_guided + (1 - g_y) * v_uniform
        z2     = a + v_gate
        logits = classifier(z2)

    Gate modes:
      * ``off``:       original fully guided AVCIL (g = 1)
      * ``learnable``: one trainable scalar per semantic class
      * ``fixed``:     a frozen gate table exported by the full-class job

    This is an oracle experiment.  When the gate is active, labels are
    intentionally required at train/validation/test time.
    """

    def __init__(self, args, step_out_class_num: int, LSC: bool = False):
        super().__init__(args, step_out_class_num, LSC=LSC)

        self.oracle_gate_mode = str(
            getattr(args, "oracle_gate_mode", "off")
        ).lower()
        if self.oracle_gate_mode not in {"off", "learnable", "fixed"}:
            raise ValueError(
                "oracle_gate_mode must be one of: off, learnable, fixed"
            )

        table_size = int(
            getattr(
                args,
                "oracle_gate_table_size",
                getattr(args, "num_classes", step_out_class_num),
            )
        )
        if table_size < step_out_class_num:
            raise ValueError(
                "oracle_gate_table_size ({}) must be >= current output classes ({})".format(
                    table_size, step_out_class_num
                )
            )

        self.oracle_gate_min = float(getattr(args, "oracle_gate_min", 0.0))
        self.oracle_gate_max = float(getattr(args, "oracle_gate_max", 1.0))
        if not (0.0 <= self.oracle_gate_min < self.oracle_gate_max <= 1.0):
            raise ValueError(
                "Require 0 <= oracle_gate_min < oracle_gate_max <= 1"
            )

        init = float(getattr(args, "oracle_gate_init", 1.0))
        init = min(max(init, self.oracle_gate_min), self.oracle_gate_max)
        self.oracle_class_gate = nn.Parameter(
            torch.full((table_size,), init, dtype=torch.float32),
            requires_grad=(self.oracle_gate_mode == "learnable"),
        )

    # ------------------------------------------------------------------
    # Gate-table utilities
    # ------------------------------------------------------------------
    def set_oracle_gate_mode(self, mode: str) -> None:
        mode = str(mode).lower()
        if mode not in {"off", "learnable", "fixed"}:
            raise ValueError("mode must be one of: off, learnable, fixed")
        self.oracle_gate_mode = mode
        self.oracle_class_gate.requires_grad_(mode == "learnable")

    def set_oracle_gate_values(
        self, values: torch.Tensor, mode: str = "fixed"
    ) -> None:
        values = torch.as_tensor(
            values, dtype=self.oracle_class_gate.dtype
        ).flatten()
        if values.numel() != self.oracle_class_gate.numel():
            raise ValueError(
                "Gate-table size mismatch: model={}, provided={}".format(
                    self.oracle_class_gate.numel(), values.numel()
                )
            )
        if not torch.isfinite(values).all():
            raise ValueError("Gate table contains NaN or Inf")

        with torch.no_grad():
            self.oracle_class_gate.copy_(
                values.clamp(self.oracle_gate_min, self.oracle_gate_max).to(
                    self.oracle_class_gate.device
                )
            )
        self.set_oracle_gate_mode(mode)

    def get_oracle_gate_values(self) -> torch.Tensor:
        return self.oracle_class_gate.clamp(
            self.oracle_gate_min, self.oracle_gate_max
        )

    @torch.no_grad()
    def project_oracle_gate_(self) -> None:
        self.oracle_class_gate.clamp_(
            self.oracle_gate_min, self.oracle_gate_max
        )

    def _sample_gate(
        self,
        labels: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if labels is None:
            raise ValueError(
                "labels are required while the O1 oracle class gate is active"
            )
        labels = labels.to(device=device, dtype=torch.long).view(-1)
        if labels.numel() == 0:
            raise ValueError("labels are empty")
        if int(labels.min().item()) < 0:
            raise IndexError("labels contain a negative class id")
        if int(labels.max().item()) >= self.oracle_class_gate.numel():
            raise IndexError(
                "label {} exceeds oracle gate-table size {}".format(
                    int(labels.max().item()), self.oracle_class_gate.numel()
                )
            )

        gates = self.get_oracle_gate_values().to(device=device, dtype=dtype)
        return gates.index_select(0, labels)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        visual=None,
        audio=None,
        labels: Optional[torch.Tensor] = None,
        use_oracle_gate: Optional[bool] = None,
        out_logits=True,
        out_features=False,
        out_features_norm=False,
        out_feature_before_fusion=False,
        out_z1_projection=False,
        out_attn_score=False,
        AFC_train_out=False,
        return_dict=False,
        out_analysis_features=False,
    ):
        # The oracle intervention is only defined for the audio-visual path.
        # Delegate unimodal paths unchanged to the user's current base model.
        if self.modality != "audio-visual":
            return super().forward(
                visual=visual,
                audio=audio,
                out_logits=out_logits,
                out_features=out_features,
                out_features_norm=out_features_norm,
                out_feature_before_fusion=out_feature_before_fusion,
                out_z1_projection=out_z1_projection,
                out_attn_score=out_attn_score,
                AFC_train_out=AFC_train_out,
                return_dict=return_dict,
                out_analysis_features=out_analysis_features,
            )

        if visual is None:
            raise ValueError("input frames are None when modality contains visual")
        if audio is None:
            raise ValueError("input audio are None when modality contains audio")

        # Same visual token layout and guided attention as the current AVCIL.
        visual_4d = visual.view(visual.shape[0], 8, -1, 768)
        visual_z0_uniform = visual_4d.mean(dim=(1, 2))

        spatial_attn_score, temporal_attn_score = self.audio_visual_attention(
            audio, visual_4d
        )

        visual_pooled_feature = torch.sum(
            spatial_attn_score * visual_4d, dim=2
        )
        visual_pooled_feature = torch.sum(
            temporal_attn_score * visual_pooled_feature, dim=1
        )

        audio_feature = F.relu(self.audio_proj(audio))
        visual_guided_feature = F.relu(
            self.visual_proj(visual_pooled_feature)
        )
        visual_uniform_feature = F.relu(
            self.visual_proj(visual_z0_uniform)
        )

        gate_active = (
            self.oracle_gate_mode != "off"
            if use_oracle_gate is None
            else bool(use_oracle_gate)
        )
        if gate_active:
            gate = self._sample_gate(
                labels,
                visual_guided_feature.device,
                visual_guided_feature.dtype,
            )
        else:
            gate = torch.ones(
                visual_guided_feature.shape[0],
                device=visual_guided_feature.device,
                dtype=visual_guided_feature.dtype,
            )

        visual_gated_feature = (
            gate.unsqueeze(1) * visual_guided_feature
            + (1.0 - gate.unsqueeze(1)) * visual_uniform_feature
        )

        audio_visual_features = visual_gated_feature + audio_feature
        logits = self.classifier(audio_visual_features)

        # -------------------------------------------------------------
        # Dict output branch: preserve all current keys and add oracle keys.
        # -------------------------------------------------------------
        if return_dict:
            outputs_dict = {}

            if AFC_train_out:
                audio_feature.retain_grad()
                visual_guided_feature.retain_grad()
                visual_pooled_feature.retain_grad()

                outputs_dict["logits"] = logits
                outputs_dict["visual_pooled_feature"] = visual_pooled_feature
                outputs_dict["audio_feature"] = audio_feature
                # Preserve the original public meaning: guided z1 visual.
                outputs_dict["visual_feature"] = visual_guided_feature
                outputs_dict["visual_uniform_feature"] = visual_uniform_feature
                outputs_dict["visual_gated_feature"] = visual_gated_feature
                outputs_dict["oracle_gate"] = gate
                return outputs_dict

            if out_logits:
                outputs_dict["logits"] = logits

            if out_features:
                if out_features_norm:
                    outputs_dict["features"] = F.normalize(
                        audio_visual_features, dim=1
                    )
                else:
                    outputs_dict["features"] = audio_visual_features

            if out_feature_before_fusion:
                # Existing AVCIL contrastive losses remain on the original
                # full-guided z1 pair, not on the oracle mixture.
                outputs_dict["audio_feature"] = F.normalize(
                    audio_feature, dim=1
                )
                outputs_dict["visual_feature"] = F.normalize(
                    visual_guided_feature, dim=1
                )

            if out_analysis_features:
                outputs_dict["z0_audio_raw"] = audio
                outputs_dict["z0_visual_uniform_raw"] = visual_z0_uniform
                outputs_dict["z0_audio_norm"] = F.normalize(audio, dim=1)
                outputs_dict["z0_visual_uniform_norm"] = F.normalize(
                    visual_z0_uniform, dim=1
                )

                outputs_dict["attn_visual_pooled_raw"] = visual_pooled_feature
                outputs_dict["attn_visual_pooled_norm"] = F.normalize(
                    visual_pooled_feature, dim=1
                )

                # Preserve current baseline names where possible.
                outputs_dict["z1_audio_raw"] = audio_feature
                outputs_dict["z1_visual_raw"] = visual_guided_feature
                outputs_dict["z1_audio_norm"] = F.normalize(
                    audio_feature, dim=1
                )
                outputs_dict["z1_visual_norm"] = F.normalize(
                    visual_guided_feature, dim=1
                )

                # Oracle-specific endpoints and actual classifier input.
                outputs_dict[
                    "z1_visual_guided_raw"
                ] = visual_guided_feature
                outputs_dict[
                    "z1_visual_uniform_raw"
                ] = visual_uniform_feature
                outputs_dict[
                    "z1_visual_gated_raw"
                ] = visual_gated_feature
                outputs_dict[
                    "z1_visual_guided_norm"
                ] = F.normalize(visual_guided_feature, dim=1)
                outputs_dict[
                    "z1_visual_uniform_norm"
                ] = F.normalize(visual_uniform_feature, dim=1)
                outputs_dict[
                    "z1_visual_gated_norm"
                ] = F.normalize(visual_gated_feature, dim=1)

                outputs_dict["z2_fusion_raw"] = audio_visual_features
                outputs_dict["z2_fusion_norm"] = F.normalize(
                    audio_visual_features, dim=1
                )
                outputs_dict["z2_fusion_guided_raw"] = (
                    audio_feature + visual_guided_feature
                )
                outputs_dict["z2_fusion_uniform_raw"] = (
                    audio_feature + visual_uniform_feature
                )
                outputs_dict["oracle_gate"] = gate

            if out_z1_projection:
                if not getattr(self, "use_z1_cm_projection_head", False):
                    raise ValueError(
                        "out_z1_projection=True but z1_cm_projection_head is not enabled"
                    )
                audio_z1_proj, visual_z1_proj = self.project_z1_features(
                    audio_feature, visual_guided_feature
                )
                outputs_dict["audio_z1_proj"] = audio_z1_proj
                outputs_dict["visual_z1_proj"] = visual_z1_proj

            if out_attn_score:
                outputs_dict["spatial_attn_score"] = spatial_attn_score
                outputs_dict["temporal_attn_score"] = temporal_attn_score

            return outputs_dict

        # -------------------------------------------------------------
        # Original tuple/tensor output branch
        # -------------------------------------------------------------
        outputs = ()

        if AFC_train_out:
            audio_feature.retain_grad()
            visual_guided_feature.retain_grad()
            visual_pooled_feature.retain_grad()
            outputs += (
                logits,
                visual_pooled_feature,
                audio_feature,
                visual_guided_feature,
            )
            return outputs

        if out_logits:
            outputs += (logits,)

        if out_features:
            if out_features_norm:
                outputs += (F.normalize(audio_visual_features, dim=1),)
            else:
                outputs += (audio_visual_features,)

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

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def incremental_classifier(self, numclass):
        # Reuse the current model's exact classifier expansion behavior.
        super().incremental_classifier(numclass)
        self.num_classes = numclass

    def extract_joint_feature(
        self,
        visual=None,
        audio=None,
        labels: Optional[torch.Tensor] = None,
        use_oracle_gate: Optional[bool] = None,
    ):
        """Return the actual gated fused feature used by the oracle model."""
        return self.forward(
            visual=visual,
            audio=audio,
            labels=labels,
            use_oracle_gate=use_oracle_gate,
            out_logits=False,
            out_features=True,
        )