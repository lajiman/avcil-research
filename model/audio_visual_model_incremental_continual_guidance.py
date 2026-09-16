import torch
import torch.nn.functional as F

from .audio_visual_model_incremental import IncreAudioVisualNet as BaselineIncreAudioVisualNet


class IncreAudioVisualNet(BaselineIncreAudioVisualNet):
    """
    AVCIL network with an explicit class-wise continual audio-to-visual gate.

    The stored latent state is an interference energy E_c >= 0. Its class gate is

        g_c = exp(-E_c).

    At inference time the class is unknown, so the sample gate is a stop-gradient
    mixture of class gates, using logits from the unguided (uniform-visual) branch:

        g(x) = sum_c p(c | a, v_uniform) g_c.

    No learnable router or MLP gate is introduced.
    """

    def __init__(self, args, step_out_class_num, LSC=False):
        super().__init__(args, step_out_class_num, LSC=LSC)

        self.use_continual_guidance = bool(getattr(args, "continual_guidance", False))
        self.guidance_routing_temperature = float(
            getattr(args, "guidance_routing_temperature", 1.0)
        )
        self.guidance_gate_min = float(getattr(args, "guidance_gate_min", 0.0))

        if self.modality == "audio-visual":
            # These buffers are saved inside every model checkpoint.
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
            self.register_buffer(
                "guidance_prior_weight",
                torch.zeros(step_out_class_num, dtype=torch.float32),
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

            # At the beginning of a new step, all old classes have a prior.
            # The Kalman update later replaces 1.0 with class-specific (1-K_c).
            self.guidance_prior_weight[:old_num_classes] = 1.0
            self.guidance_prior_weight[old_num_classes:] = 0.0

    def class_guidance_gates(self):
        gates = torch.exp(-self.guidance_energy)
        return gates.clamp(min=self.guidance_gate_min, max=1.0)

    def prior_weights_for_labels(self, labels):
        labels = labels.long().to(self.guidance_prior_weight.device)
        return self.guidance_prior_weight.index_select(0, labels)

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
        tensors = {
            "energy": energy,
            "variance": variance,
            "z2_difficulty": z2_difficulty,
            "initialized": initialized,
            "prior_weight": prior_weight,
        }
        for name, value in tensors.items():
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

    def _sample_guidance_gate(self, audio_feature, visual_uniform_feature):
        routing_feature = audio_feature + visual_uniform_feature
        routing_logits = self.classifier(routing_feature)
        routing_probs = F.softmax(
            routing_logits / max(self.guidance_routing_temperature, 1e-6), dim=1
        ).detach()

        if self.use_continual_guidance:
            class_gates = self.class_guidance_gates().to(routing_probs.dtype)
            sample_gate = torch.matmul(routing_probs, class_gates)
        else:
            sample_gate = torch.ones(
                routing_probs.shape[0],
                device=routing_probs.device,
                dtype=routing_probs.dtype,
            )

        return sample_gate.clamp(0.0, 1.0), routing_logits, routing_probs

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
        # Preserve the baseline behavior for unimodal branches.
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

        sample_gate, routing_logits, routing_probs = self._sample_guidance_gate(
            audio_feature, visual_uniform_feature
        )
        gate_2d = sample_gate.unsqueeze(1)

        # Exact z1 interpolation used by the classifier.
        visual_feature = (
            gate_2d * visual_guided_feature
            + (1.0 - gate_2d) * visual_uniform_feature
        )
        audio_visual_features = visual_feature + audio_feature
        logits = self.classifier(audio_visual_features)

        # Effective maps used by attention distillation. Because the original
        # attention is factorized over space and time, these maps are the
        # corresponding explicit gate interpolation for each factor.
        spatial_uniform = torch.full_like(
            spatial_attn_raw, 1.0 / spatial_attn_raw.shape[2]
        )
        temporal_uniform = torch.full_like(
            temporal_attn_raw, 1.0 / temporal_attn_raw.shape[1]
        )
        spatial_gate = sample_gate.view(-1, 1, 1, 1)
        temporal_gate = sample_gate.view(-1, 1, 1)
        spatial_attn_effective = (
            spatial_gate * spatial_attn_raw
            + (1.0 - spatial_gate) * spatial_uniform
        )
        temporal_attn_effective = (
            temporal_gate * temporal_attn_raw
            + (1.0 - temporal_gate) * temporal_uniform
        )

        if return_dict:
            outputs_dict = {}

            if AFC_train_out:
                audio_feature.retain_grad()
                visual_feature.retain_grad()
                visual_guided_pooled_raw.retain_grad()
                outputs_dict["logits"] = logits
                outputs_dict["visual_pooled_feature"] = visual_guided_pooled_raw
                outputs_dict["audio_feature"] = audio_feature
                outputs_dict["visual_feature"] = visual_feature
                return outputs_dict

            if out_logits:
                outputs_dict["logits"] = logits

            if out_features:
                outputs_dict["features"] = (
                    F.normalize(audio_visual_features, dim=1)
                    if out_features_norm
                    else audio_visual_features
                )

            if out_feature_before_fusion:
                outputs_dict["audio_feature"] = F.normalize(
                    audio_feature, dim=1
                )
                outputs_dict["visual_feature"] = F.normalize(
                    visual_feature, dim=1
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
                outputs_dict["z1_visual_raw"] = visual_feature
                outputs_dict["z2_fusion_raw"] = audio_visual_features

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
                    visual_feature, dim=1
                )
                outputs_dict["z2_fusion_norm"] = F.normalize(
                    audio_visual_features, dim=1
                )

                outputs_dict["sample_guidance_gate"] = sample_gate
                outputs_dict["routing_logits"] = routing_logits
                outputs_dict["routing_probs"] = routing_probs
                outputs_dict["class_guidance_gates"] = self.class_guidance_gates()

            if out_z1_projection:
                if not getattr(self, "use_z1_cm_projection_head", False):
                    raise ValueError(
                        "out_z1_projection=True but z1_cm_projection_head is not enabled"
                    )
                audio_z1_proj, visual_z1_proj = self.project_z1_features(
                    audio_feature, visual_feature
                )
                outputs_dict["audio_z1_proj"] = audio_z1_proj
                outputs_dict["visual_z1_proj"] = visual_z1_proj

            if out_attn_score:
                outputs_dict["spatial_attn_score"] = spatial_attn_effective
                outputs_dict["temporal_attn_score"] = temporal_attn_effective
                outputs_dict["spatial_attn_score_raw"] = spatial_attn_raw
                outputs_dict["temporal_attn_score_raw"] = temporal_attn_raw

            return outputs_dict

        outputs = ()

        if AFC_train_out:
            audio_feature.retain_grad()
            visual_feature.retain_grad()
            visual_guided_pooled_raw.retain_grad()
            return (
                logits,
                visual_guided_pooled_raw,
                audio_feature,
                visual_feature,
            )

        if out_logits:
            outputs += (logits,)

        if out_features:
            outputs += (
                F.normalize(audio_visual_features, dim=1)
                if out_features_norm
                else audio_visual_features,
            )

        if out_feature_before_fusion:
            outputs += (
                F.normalize(audio_feature, dim=1),
                F.normalize(visual_feature, dim=1),
            )

        if out_z1_projection:
            if not getattr(self, "use_z1_cm_projection_head", False):
                raise ValueError(
                    "out_z1_projection=True but z1_cm_projection_head is not enabled"
                )
            audio_z1_proj, visual_z1_proj = self.project_z1_features(
                audio_feature, visual_feature
            )
            outputs += (audio_z1_proj, visual_z1_proj)

        if out_attn_score:
            outputs += (spatial_attn_effective, temporal_attn_effective)

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def extract_joint_feature(self, visual=None, audio=None):
        if self.modality != "audio-visual":
            return super().extract_joint_feature(visual=visual, audio=audio)

        result = self.forward(
            visual=visual,
            audio=audio,
            out_logits=False,
            out_features=True,
            return_dict=True,
        )
        return result["features"]