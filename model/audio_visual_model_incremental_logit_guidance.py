import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_visual_model_incremental import IncreAudioVisualNet as _BaseIncreAudioVisualNet


class IncreAudioVisualNet(_BaseIncreAudioVisualNet):
    """
    Full AVCIL model + explicit class-wise logit guidance gate.

    Core intervention:
        logits_G = classifier(z1_audio + z1_visual_guided)
        logits_U = classifier(z1_audio + z1_visual_uniform)

        logits[:, c] =
            g_c * logits_G[:, c] + (1 - g_c) * logits_U[:, c]

    where g_c is a class-wise binary gate estimated from the direct
    target-vs-rest decision contribution of audio-guided visual attention.

    Important:
      * No geometry is used.
      * No MLP/sample router is used.
      * Raw audio-guided z1 features are still returned for the original
        instance/class contrastive losses.
      * Raw spatial/temporal attention is still returned for the original VAD.
      * If guidance is disabled, or all gates are exactly one, forward falls
        back to the original baseline forward path.
    """

    def __init__(self, args, step_out_class_num, LSC=False):
        super().__init__(args, step_out_class_num, LSC=LSC)
        self.register_buffer(
            "logit_guidance_gates",
            torch.ones(step_out_class_num, dtype=torch.float32),
        )
        self.logit_guidance_enabled = False

    def set_logit_guidance_enabled(self, enabled: bool):
        self.logit_guidance_enabled = bool(enabled)

    def get_class_gates(self):
        return self.logit_guidance_gates.detach().clone()

    @torch.no_grad()
    def set_class_gates(self, gates, class_ids=None):
        gates = torch.as_tensor(
            gates,
            dtype=self.logit_guidance_gates.dtype,
            device=self.logit_guidance_gates.device,
        )
        if class_ids is None:
            if gates.numel() != self.logit_guidance_gates.numel():
                raise ValueError(
                    "Gate length mismatch: got {}, expected {}".format(
                        gates.numel(), self.logit_guidance_gates.numel()
                    )
                )
            self.logit_guidance_gates.copy_(gates.view_as(self.logit_guidance_gates))
            return

        class_ids = torch.as_tensor(
            class_ids, dtype=torch.long, device=self.logit_guidance_gates.device
        )
        if gates.numel() != class_ids.numel():
            raise ValueError("gates and class_ids must have the same length")
        if class_ids.numel() > 0:
            if class_ids.min().item() < 0 or class_ids.max().item() >= self.logit_guidance_gates.numel():
                raise ValueError("class_ids are out of range")
            self.logit_guidance_gates[class_ids] = gates

    def incremental_classifier(self, numclass):
        old_num = int(self.logit_guidance_gates.numel())
        old_gates = self.logit_guidance_gates.detach().clone()

        super().incremental_classifier(numclass)

        new_gates = torch.ones(
            numclass,
            dtype=old_gates.dtype,
            device=old_gates.device,
        )
        new_gates[:old_num] = old_gates
        self.logit_guidance_gates = new_gates
        self.num_classes = numclass

    def _compute_guided_uniform_branches(self, visual, audio):
        if self.modality != "audio-visual":
            raise ValueError("Guidance branches are only defined for audio-visual modality")
        if visual is None or audio is None:
            raise ValueError("visual and audio must both be provided")

        visual_4d = visual.view(visual.shape[0], 8, -1, 768)

        # Original baseline audio-guided attention branch.
        spatial_attn_score, temporal_attn_score = self.audio_visual_attention(
            audio, visual_4d
        )
        guided_pool = torch.sum(spatial_attn_score * visual_4d, dim=2)
        guided_pool = torch.sum(temporal_attn_score * guided_pool, dim=1)

        # Controlled unguided branch: uniform average over exactly the same
        # spatiotemporal visual tokens. Audio branch remains unchanged.
        uniform_pool = visual_4d.mean(dim=(1, 2))

        audio_feature = F.relu(self.audio_proj(audio))
        visual_guided_feature = F.relu(self.visual_proj(guided_pool))
        visual_uniform_feature = F.relu(self.visual_proj(uniform_pool))

        guided_fused = audio_feature + visual_guided_feature
        uniform_fused = audio_feature + visual_uniform_feature

        logits_guided = self.classifier(guided_fused)
        logits_uniform = self.classifier(uniform_fused)

        return {
            "visual_4d": visual_4d,
            "spatial_attn_score": spatial_attn_score,
            "temporal_attn_score": temporal_attn_score,
            "guided_pool": guided_pool,
            "uniform_pool": uniform_pool,
            "audio_feature": audio_feature,
            "visual_guided_feature": visual_guided_feature,
            "visual_uniform_feature": visual_uniform_feature,
            "guided_fused": guided_fused,
            "uniform_fused": uniform_fused,
            "logits_guided": logits_guided,
            "logits_uniform": logits_uniform,
        }

    def guided_uniform_logits(self, visual=None, audio=None):
        """
        Forced endpoint logits used to estimate guidance contribution.
        This ignores the currently active gate by construction.
        """
        out = self._compute_guided_uniform_branches(visual, audio)
        return out["logits_guided"], out["logits_uniform"]

    def _routed_logits(self, logits_guided, logits_uniform):
        num_out = logits_guided.shape[1]
        if self.logit_guidance_gates.numel() < num_out:
            raise RuntimeError(
                "Guidance gate buffer has {} classes but classifier has {}".format(
                    self.logit_guidance_gates.numel(), num_out
                )
            )

        gates = self.logit_guidance_gates[:num_out].to(
            device=logits_guided.device, dtype=logits_guided.dtype
        ).view(1, -1)

        # Exact endpoint interpolation in decision space.
        return gates * logits_guided + (1.0 - gates) * logits_uniform

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
        # Preserve the original code path exactly whenever guidance is inactive.
        if self.modality != "audio-visual" or not self.logit_guidance_enabled:
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

        # If every visible class uses g=1, also use the exact baseline path.
        num_out = self.classifier.out_features
        if torch.all(self.logit_guidance_gates[:num_out] == 1):
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

        branch = self._compute_guided_uniform_branches(visual, audio)
        audio_feature = branch["audio_feature"]
        visual_feature = branch["visual_guided_feature"]
        visual_pooled_feature = branch["guided_pool"]
        guided_fused = branch["guided_fused"]
        logits = self._routed_logits(
            branch["logits_guided"], branch["logits_uniform"]
        )

        spatial_attn_score = branch["spatial_attn_score"]
        temporal_attn_score = branch["temporal_attn_score"]

        # The original AVCIL auxiliary losses remain attached to the original
        # guided z1 branch. Only classification logits are routed.
        if return_dict:
            outputs_dict = {}

            if AFC_train_out:
                audio_feature.retain_grad()
                visual_feature.retain_grad()
                visual_pooled_feature.retain_grad()
                outputs_dict["logits"] = logits
                outputs_dict["visual_pooled_feature"] = visual_pooled_feature
                outputs_dict["audio_feature"] = audio_feature
                outputs_dict["visual_feature"] = visual_feature
                return outputs_dict

            if out_logits:
                outputs_dict["logits"] = logits

            if out_features:
                outputs_dict["features"] = (
                    F.normalize(guided_fused, dim=1)
                    if out_features_norm else guided_fused
                )

            if out_feature_before_fusion:
                outputs_dict["audio_feature"] = F.normalize(audio_feature, dim=1)
                outputs_dict["visual_feature"] = F.normalize(visual_feature, dim=1)

            if out_analysis_features:
                visual_z0_uniform = branch["visual_4d"].mean(dim=(1, 2))
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
                outputs_dict["z1_audio_raw"] = audio_feature
                outputs_dict["z1_visual_raw"] = visual_feature
                outputs_dict["z1_visual_uniform_raw"] = branch["visual_uniform_feature"]
                outputs_dict["z2_fusion_raw"] = guided_fused
                outputs_dict["z1_audio_norm"] = F.normalize(audio_feature, dim=1)
                outputs_dict["z1_visual_norm"] = F.normalize(visual_feature, dim=1)
                outputs_dict["z1_visual_uniform_norm"] = F.normalize(
                    branch["visual_uniform_feature"], dim=1
                )
                outputs_dict["z2_fusion_norm"] = F.normalize(guided_fused, dim=1)
                outputs_dict["logits_guided"] = branch["logits_guided"]
                outputs_dict["logits_uniform"] = branch["logits_uniform"]
                outputs_dict["class_guidance_gates"] = self.logit_guidance_gates[
                    :logits.shape[1]
                ]

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
                outputs_dict["spatial_attn_score"] = spatial_attn_score
                outputs_dict["temporal_attn_score"] = temporal_attn_score

            return outputs_dict

        outputs = ()

        if AFC_train_out:
            audio_feature.retain_grad()
            visual_feature.retain_grad()
            visual_pooled_feature.retain_grad()
            return logits, visual_pooled_feature, audio_feature, visual_feature

        if out_logits:
            outputs += (logits,)

        if out_features:
            outputs += (
                F.normalize(guided_fused, dim=1)
                if out_features_norm else guided_fused,
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
            outputs += (spatial_attn_score, temporal_attn_score)

        if len(outputs) == 1:
            return outputs[0]
        return outputs