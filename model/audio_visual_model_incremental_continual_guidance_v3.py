import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_visual_model_incremental import IncreAudioVisualNet as BaselineIncreAudioVisualNet


class IncreAudioVisualNet(BaselineIncreAudioVisualNet):
    """
    AVCIL with an explicit class-wise continual audio-to-visual guidance prior.

    Version 4 keeps the model-side operation deliberately small: the network
    stores only the *active* class-wise posterior used for the current epoch.
    The frozen previous-step prior and periodic re-estimation logic live in the
    training script, so repeated refreshes within one incremental step cannot
    accidentally become recursive Kalman updates.

    For class c, the stored interference energy E_c defines

        g_c = exp(-E_c),  0 < g_c <= 1.

    Because the classifier is linear, the class-wise gate is applied exactly:

        logit_c = w_c^T a
                  + g_c w_c^T v_guided
                  + (1-g_c) w_c^T v_uniform
                  + b_c.

    No label, pseudo-label, learned router, or probability-weighted mixture of
    class gates is required at inference time.
    """

    def __init__(self, args, step_out_class_num, LSC=False):
        super().__init__(args, step_out_class_num, LSC=LSC)

        self.use_continual_guidance = bool(
            getattr(args, "continual_guidance", False)
        )
        self.guidance_gate_min = float(getattr(args, "guidance_gate_min", 0.0))

        if self.modality == "audio-visual":
            self.register_buffer(
                "guidance_energy",
                torch.zeros(step_out_class_num, dtype=torch.float32),
            )
            self.register_buffer(
                "guidance_variance",
                torch.ones(step_out_class_num, dtype=torch.float32),
            )
            self.register_buffer(
                "guidance_z2_difficulty",
                torch.zeros(step_out_class_num, dtype=torch.float32),
            )
            self.register_buffer(
                "guidance_initialized",
                torch.zeros(step_out_class_num, dtype=torch.bool),
            )
            # eta_c = 1-K_c; used to weight raw attention distillation.
            self.register_buffer(
                "guidance_prior_weight",
                torch.zeros(step_out_class_num, dtype=torch.float32),
            )
            # The initial task must be trainable with the exact baseline
            # computational graph. This scalar controls whether class-wise
            # routing is active in ordinary forward passes.
            self.register_buffer(
                "guidance_forward_active",
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
        old_num_classes = int(self.classifier.out_features)
        super().incremental_classifier(numclass)
        self.num_classes = numclass

        if self.modality == "audio-visual" and hasattr(self, "guidance_energy"):
            self.guidance_energy = self._resize_1d_buffer(
                self.guidance_energy, numclass, 0.0
            )
            self.guidance_variance = self._resize_1d_buffer(
                self.guidance_variance, numclass, 1.0
            )
            self.guidance_z2_difficulty = self._resize_1d_buffer(
                self.guidance_z2_difficulty, numclass, 0.0
            )
            self.guidance_initialized = self._resize_1d_buffer(
                self.guidance_initialized, numclass, False
            ).bool()
            self.guidance_prior_weight = self._resize_1d_buffer(
                self.guidance_prior_weight, numclass, 0.0
            )

            self.guidance_prior_weight[:old_num_classes] = 1.0
            self.guidance_prior_weight[old_num_classes:] = 0.0

    def is_guidance_forward_active(self):
        return bool(self.guidance_forward_active.item())

    @torch.no_grad()
    def set_guidance_forward_active(self, active):
        self.guidance_forward_active.fill_(bool(active))

    def class_guidance_gates(self):
        if (
            not self.use_continual_guidance
            or not self.is_guidance_forward_active()
        ):
            return torch.ones_like(self.guidance_energy)
        return torch.exp(-self.guidance_energy).clamp(
            min=self.guidance_gate_min, max=1.0
        )

    def prior_weights_for_labels(self, labels):
        labels = labels.long().to(self.guidance_prior_weight.device)
        if not self.is_guidance_forward_active():
            return torch.ones(
                labels.shape[0],
                device=self.guidance_prior_weight.device,
                dtype=self.guidance_prior_weight.dtype,
            )
        return self.guidance_prior_weight.index_select(0, labels)

    @torch.no_grad()
    def export_guidance_state(self):
        """Return a detached copy of the active class-wise state."""
        return {
            "energy": self.guidance_energy.detach().clone(),
            "variance": self.guidance_variance.detach().clone(),
            "z2_difficulty": self.guidance_z2_difficulty.detach().clone(),
            "initialized": self.guidance_initialized.detach().clone(),
            "prior_weight": self.guidance_prior_weight.detach().clone(),
        }

    @torch.no_grad()
    def set_guidance_state(
        self,
        energy,
        variance,
        z2_difficulty,
        initialized,
        prior_weight,
    ):
        expected = self.classifier.out_features
        values = {
            "energy": energy,
            "variance": variance,
            "z2_difficulty": z2_difficulty,
            "initialized": initialized,
            "prior_weight": prior_weight,
        }
        for name, value in values.items():
            if value.numel() != expected:
                raise ValueError(
                    "{} has {} entries, expected {}".format(
                        name, value.numel(), expected
                    )
                )

        self.guidance_energy.copy_(energy.to(self.guidance_energy.device))
        self.guidance_variance.copy_(variance.to(self.guidance_variance.device))
        self.guidance_z2_difficulty.copy_(
            z2_difficulty.to(self.guidance_z2_difficulty.device)
        )
        self.guidance_initialized.copy_(
            initialized.to(self.guidance_initialized.device).bool()
        )
        self.guidance_prior_weight.copy_(
            prior_weight.to(self.guidance_prior_weight.device)
        )

    def _classwise_guided_logits(
        self,
        audio_feature,
        visual_guided_feature,
        visual_uniform_feature,
    ):
        """Exact class-wise routing for a linear classifier."""
        if not isinstance(self.classifier, nn.Linear):
            raise TypeError(
                "Continual guidance requires nn.Linear because its exact "
                "derivation uses classifier linearity."
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

        class_gates = self.class_guidance_gates().to(
            device=audio_logits.device, dtype=audio_logits.dtype
        )
        logits = (
            audio_logits
            + class_gates.unsqueeze(0) * guided_visual_logits
            + (1.0 - class_gates.unsqueeze(0)) * uniform_visual_logits
        )
        if bias is not None:
            logits = logits + bias.unsqueeze(0)

        return (
            logits,
            audio_logits,
            guided_visual_logits,
            uniform_visual_logits,
            class_gates,
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

        # When guidance is disabled (in particular at incremental step 0),
        # ordinary training/validation/test calls use the parent implementation
        # verbatim. This is stronger than merely setting g_c=1: it preserves
        # the exact baseline floating-point operation order and gradients.
        # The sole exception is the analysis-only call used to estimate the
        # consolidated prior after the best checkpoint has been selected.
        analysis_only_custom_path = bool(return_dict and out_analysis_features)
        if (
            not self.is_guidance_forward_active()
            and not analysis_only_custom_path
        ):
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

        # Audio-independent reference branch.
        visual_uniform_pooled_raw = visual_4d.mean(dim=(1, 2))

        # Original AVCIL audio-guided branch.
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
        ) = self._classwise_guided_logits(
            audio_feature,
            visual_guided_feature,
            visual_uniform_feature,
        )

        # A class-wise logit gate does not induce one class-independent mixed
        # z1/z2 feature. The original guided branch is therefore the reference
        # used for z1 contrastive learning and z2 geometry.
        visual_feature_reference = visual_guided_feature
        z2_guided_reference = audio_feature + visual_guided_feature
        z2_uniform_reference = audio_feature + visual_uniform_feature

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
                outputs_dict["z0_audio_raw"] = audio
                outputs_dict["z0_visual_uniform_raw"] = visual_uniform_pooled_raw
                outputs_dict["z0_audio_norm"] = F.normalize(audio, dim=1)
                outputs_dict["z0_visual_uniform_norm"] = F.normalize(
                    visual_uniform_pooled_raw, dim=1
                )

                outputs_dict["attn_visual_pooled_raw"] = visual_guided_pooled_raw
                outputs_dict["attn_visual_pooled_norm"] = F.normalize(
                    visual_guided_pooled_raw, dim=1
                )

                outputs_dict["z1_audio_raw"] = audio_feature
                outputs_dict["z1_visual_guided_raw"] = visual_guided_feature
                outputs_dict["z1_visual_uniform_raw"] = visual_uniform_feature
                outputs_dict["z1_visual_raw"] = visual_feature_reference

                outputs_dict["z1_audio_norm"] = F.normalize(
                    audio_feature, dim=1
                )
                outputs_dict["z1_visual_guided_norm"] = F.normalize(
                    visual_guided_feature, dim=1
                )
                outputs_dict["z1_visual_uniform_norm"] = F.normalize(
                    visual_uniform_feature, dim=1
                )
                outputs_dict["z1_visual_norm"] = F.normalize(
                    visual_feature_reference, dim=1
                )

                outputs_dict["z2_guided_reference_raw"] = z2_guided_reference
                outputs_dict["z2_guided_reference_norm"] = F.normalize(
                    z2_guided_reference, dim=1
                )
                outputs_dict["z2_uniform_reference_raw"] = z2_uniform_reference
                outputs_dict["z2_uniform_reference_norm"] = F.normalize(
                    z2_uniform_reference, dim=1
                )
                # Backward-compatible keys, deliberately gate-independent.
                outputs_dict["z2_fusion_raw"] = z2_guided_reference
                outputs_dict["z2_fusion_norm"] = F.normalize(
                    z2_guided_reference, dim=1
                )

                outputs_dict["class_guidance_gates"] = class_gates
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
                # VAD distills the actual raw AVCIL attention. A class-wise
                # logit gate does not define one sample-level effective map.
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
        """
        Return the original guided z2 reference. Class-wise routing is applied
        only to logits, so no unique class-independent mixed feature exists.
        """
        if (
            self.modality != "audio-visual"
            or not self.is_guidance_forward_active()
        ):
            return super().extract_joint_feature(visual=visual, audio=audio)

        if visual is None or audio is None:
            raise ValueError("visual and audio inputs are required")

        visual_4d = visual.view(visual.shape[0], 8, -1, 768)
        spatial_attn, temporal_attn = self.audio_visual_attention(
            audio, visual_4d
        )
        visual_pooled = torch.sum(spatial_attn * visual_4d, dim=2)
        visual_pooled = torch.sum(temporal_attn * visual_pooled, dim=1)
        audio_feature = F.relu(self.audio_proj(audio))
        visual_feature = F.relu(self.visual_proj(visual_pooled))
        return audio_feature + visual_feature