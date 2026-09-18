import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_visual_model_incremental import IncreAudioVisualNet as BaselineIncreAudioVisualNet


class IncreAudioVisualNet(BaselineIncreAudioVisualNet):
    """
    AVCIL with a current-step explicit class-wise energy gate.

    This model contains no continual-prior, Kalman, z2-drift, uncertainty, or
    cross-step state update. For every currently seen class c, the externally
    estimated energy E_c defines

        g_c = exp(-E_c),  0 < g_c <= 1.

    Because the AVCIL classifier is linear, the class-wise gate is applied
    exactly at the logit level:

        logit_c = w_c^T a
                  + g_c w_c^T v_guided
                  + (1-g_c) w_c^T v_uniform
                  + b_c.

    The original guided z1 audio/visual features are still returned for the
    existing instance- and class-level contrastive losses. Raw spatial and
    temporal attention maps are still returned for the original VAD loss.
    """

    def __init__(self, args, step_out_class_num, LSC=False):
        super().__init__(args, step_out_class_num, LSC=LSC)
        self.use_energy_gate = bool(getattr(args, "energy_gate", False))
        self.energy_gate_min = float(getattr(args, "energy_gate_min", 0.0))

        if self.modality == "audio-visual":
            self.register_buffer(
                "class_energy",
                torch.zeros(step_out_class_num, dtype=torch.float32),
            )
            self.register_buffer(
                "energy_forward_active",
                torch.tensor(False, dtype=torch.bool),
            )

    @staticmethod
    def _resize_1d_buffer(old_buffer, new_size, fill_value):
        new_buffer = old_buffer.new_full((new_size,), fill_value)
        copy_size = min(old_buffer.numel(), new_size)
        if copy_size > 0:
            new_buffer[:copy_size] = old_buffer[:copy_size]
        return new_buffer

    def incremental_classifier(self, numclass):
        super().incremental_classifier(numclass)
        self.num_classes = numclass
        if self.modality == "audio-visual" and hasattr(self, "class_energy"):
            self.class_energy = self._resize_1d_buffer(
                self.class_energy, numclass, 0.0
            )

    def is_energy_forward_active(self):
        return bool(self.energy_forward_active.item())

    @torch.no_grad()
    def set_energy_forward_active(self, active):
        self.energy_forward_active.fill_(bool(active))

    @torch.no_grad()
    def reset_energy_state(self):
        """Remove all cross-step energy carry-over and restore baseline g=1."""
        self.class_energy.zero_()
        self.energy_forward_active.fill_(False)

    @torch.no_grad()
    def set_class_energy(self, energy, activate=True):
        if energy.ndim != 1 or energy.numel() != self.classifier.out_features:
            raise ValueError(
                "energy has shape {}, expected ({},)".format(
                    tuple(energy.shape), self.classifier.out_features
                )
            )
        if not torch.isfinite(energy).all():
            raise ValueError("energy contains NaN or Inf")
        if (energy < 0).any():
            raise ValueError("energy must be non-negative")
        self.class_energy.copy_(energy.to(self.class_energy.device))
        self.energy_forward_active.fill_(bool(activate))

    def class_energy_gates(self):
        if not self.use_energy_gate or not self.is_energy_forward_active():
            return torch.ones_like(self.class_energy)
        return torch.exp(-self.class_energy).clamp(
            min=self.energy_gate_min, max=1.0
        )

    def _classwise_energy_logits(
        self,
        audio_feature,
        visual_guided_feature,
        visual_uniform_feature,
    ):
        if not isinstance(self.classifier, nn.Linear):
            raise TypeError(
                "The exact class-wise energy gate requires nn.Linear because "
                "its derivation uses classifier linearity."
            )

        weight = self.classifier.weight
        bias = self.classifier.bias

        audio_logits = F.linear(audio_feature, weight, bias=None)
        guided_visual_logits = F.linear(
            visual_guided_feature, weight, bias=None
        )
        uniform_visual_logits = F.linear(
            visual_uniform_feature, weight, bias=None
        )
        gates = self.class_energy_gates().to(
            device=audio_logits.device, dtype=audio_logits.dtype
        )

        logits = (
            audio_logits
            + gates.unsqueeze(0) * guided_visual_logits
            + (1.0 - gates.unsqueeze(0)) * uniform_visual_logits
        )
        if bias is not None:
            logits = logits + bias.unsqueeze(0)

        return (
            logits,
            audio_logits,
            guided_visual_logits,
            uniform_visual_logits,
            gates,
        )

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
    ):
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

        # During warm-up, ordinary forward calls use the parent implementation
        # verbatim. This guarantees exact baseline computation when g=1.
        analysis_only_custom_path = bool(return_dict and out_analysis_features)
        if not self.is_energy_forward_active() and not analysis_only_custom_path:
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

        visual_4d = visual.view(visual.shape[0], 8, -1, 768)
        visual_uniform_pooled_raw = visual_4d.mean(dim=(1, 2))

        spatial_attn_raw, temporal_attn_raw = self.audio_visual_attention(
            audio, visual_4d
        )
        visual_guided_pooled_raw = torch.sum(
            spatial_attn_raw * visual_4d, dim=2
        )
        visual_guided_pooled_raw = torch.sum(
            temporal_attn_raw * visual_guided_pooled_raw, dim=1
        )

        audio_feature = F.relu(self.audio_proj(audio))
        visual_guided_feature = F.relu(
            self.visual_proj(visual_guided_pooled_raw)
        )
        visual_uniform_feature = F.relu(
            self.visual_proj(visual_uniform_pooled_raw)
        )

        (
            logits,
            audio_logits,
            guided_visual_logits,
            uniform_visual_logits,
            class_gates,
        ) = self._classwise_energy_logits(
            audio_feature,
            visual_guided_feature,
            visual_uniform_feature,
        )

        # The original guided branch remains the z1 contrastive reference.
        visual_feature_reference = visual_guided_feature
        z2_guided_reference = audio_feature + visual_guided_feature

        if return_dict:
            outputs_dict = {}

            if AFC_train_out:
                audio_feature.retain_grad()
                visual_feature_reference.retain_grad()
                visual_guided_pooled_raw.retain_grad()
                outputs_dict["logits"] = logits
                outputs_dict["visual_pooled_feature"] = visual_guided_pooled_raw
                outputs_dict["audio_feature"] = audio_feature
                outputs_dict["visual_feature"] = visual_feature_reference
                return outputs_dict

            if out_logits:
                outputs_dict["logits"] = logits

            if out_features:
                outputs_dict["features"] = (
                    F.normalize(z2_guided_reference, dim=1)
                    if out_features_norm
                    else z2_guided_reference
                )

            if out_feature_before_fusion:
                outputs_dict["audio_feature"] = F.normalize(
                    audio_feature, dim=1
                )
                outputs_dict["visual_feature"] = F.normalize(
                    visual_feature_reference, dim=1
                )

            if out_analysis_features:
                outputs_dict["z1_audio_raw"] = audio_feature
                outputs_dict["z1_visual_guided_raw"] = visual_guided_feature
                outputs_dict["z1_visual_uniform_raw"] = visual_uniform_feature
                outputs_dict["z1_audio_norm"] = F.normalize(
                    audio_feature, dim=1
                )
                outputs_dict["z1_visual_guided_norm"] = F.normalize(
                    visual_guided_feature, dim=1
                )
                outputs_dict["z1_visual_uniform_norm"] = F.normalize(
                    visual_uniform_feature, dim=1
                )
                outputs_dict["class_energy"] = self.class_energy
                outputs_dict["class_energy_gates"] = class_gates
                outputs_dict["audio_logits"] = audio_logits
                outputs_dict["guided_visual_logits"] = guided_visual_logits
                outputs_dict["uniform_visual_logits"] = uniform_visual_logits

            if out_z1_projection:
                if not getattr(self, "use_z1_cm_projection_head", False):
                    raise ValueError(
                        "out_z1_projection=True but z1_cm_projection_head is not enabled"
                    )
                audio_z1_proj, visual_z1_proj = self.project_z1_features(
                    audio_feature, visual_feature_reference
                )
                outputs_dict["audio_z1_proj"] = audio_z1_proj
                outputs_dict["visual_z1_proj"] = visual_z1_proj

            if out_attn_score:
                outputs_dict["spatial_attn_score"] = spatial_attn_raw
                outputs_dict["temporal_attn_score"] = temporal_attn_raw

            return outputs_dict

        if AFC_train_out:
            audio_feature.retain_grad()
            visual_feature_reference.retain_grad()
            visual_guided_pooled_raw.retain_grad()
            return (
                logits,
                visual_guided_pooled_raw,
                audio_feature,
                visual_feature_reference,
            )

        outputs = ()
        if out_logits:
            outputs += (logits,)
        if out_features:
            outputs += (
                F.normalize(z2_guided_reference, dim=1)
                if out_features_norm
                else z2_guided_reference,
            )
        if out_feature_before_fusion:
            outputs += (
                F.normalize(audio_feature, dim=1),
                F.normalize(visual_feature_reference, dim=1),
            )
        if out_z1_projection:
            if not getattr(self, "use_z1_cm_projection_head", False):
                raise ValueError(
                    "out_z1_projection=True but z1_cm_projection_head is not enabled"
                )
            audio_z1_proj, visual_z1_proj = self.project_z1_features(
                audio_feature, visual_feature_reference
            )
            outputs += (audio_z1_proj, visual_z1_proj)
        if out_attn_score:
            outputs += (spatial_attn_raw, temporal_attn_raw)

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def extract_joint_feature(self, visual=None, audio=None):
        # No unique class-independent gated feature exists; return the original
        # guided AVCIL z2 feature exactly as the baseline stage-2 API expects.
        return super().extract_joint_feature(visual=visual, audio=audio)