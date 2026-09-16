"""
Offline class-selective uniform-visual branch for AVCIL.

Place this file at:
    model/audio_visual_model_incremental_offline_uniform.py

The model keeps the original audio-guided visual branch and adds an
parameter-free uniform visual pooling branch.  A fixed offline class gate
selects which visual branch contributes to each candidate class logit:

    gate[c] = 1 -> original audio-guided visual evidence
    gate[c] = 0 -> audio-independent uniform visual evidence

The audio branch always contributes to every class logit.  The gate is fixed,
class-wise, and does not require a ground-truth label at inference time.
"""

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_visual_model_incremental import IncreAudioVisualNet


class IncreAudioVisualNetOfflineUniform(IncreAudioVisualNet):
    """AVCIL with a fixed class-wise guided/uniform visual routing gate."""

    def __init__(self, args, step_out_class_num: int, LSC: bool = False):
        if LSC:
            raise NotImplementedError(
                "Offline class-wise logit routing currently supports nn.Linear only."
            )

        super().__init__(args, step_out_class_num, LSC=False)

        self.offline_uniform_enabled = bool(
            getattr(args, "offline_uniform", False)
        )
        self.offline_unguided_class_ids = sorted(
            {
                int(class_id)
                for class_id in getattr(
                    args, "offline_unguided_class_ids", []
                )
            }
        )

        self.register_buffer(
            "class_attention_gate",
            self._make_class_gate(step_out_class_num),
            persistent=True,
        )

    def _make_class_gate(
        self,
        num_classes: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Return a vector of length ``num_classes``.

        1 means that class uses audio-guided visual pooling.
        0 means that class uses uniform visual pooling.
        """
        gate = torch.ones(
            num_classes,
            device=device,
            dtype=dtype if dtype is not None else torch.float32,
        )

        if self.offline_uniform_enabled:
            for class_id in self.offline_unguided_class_ids:
                if 0 <= class_id < num_classes:
                    gate[class_id] = 0.0

        return gate

    def set_offline_unguided_classes(
        self, class_ids: Iterable[int]
    ) -> None:
        """Replace the fixed offline class set and refresh the current gate."""
        self.offline_unguided_class_ids = sorted(
            {int(class_id) for class_id in class_ids}
        )
        self.class_attention_gate = self._make_class_gate(
            self.classifier.out_features,
            device=self.classifier.weight.device,
            dtype=self.classifier.weight.dtype,
        )

    def has_unguided_old_class(self, num_old_classes: int) -> bool:
        """Whether any currently old class uses the uniform branch."""
        if num_old_classes <= 0:
            return False
        gate = self.class_attention_gate[:num_old_classes]
        return bool((gate < 0.5).any().item())

    def _classwise_routed_logits(
        self,
        audio_feature: torch.Tensor,
        guided_visual_feature: torch.Tensor,
        uniform_visual_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Compute label-free class-wise routed logits."""
        if not isinstance(self.classifier, nn.Linear):
            raise TypeError(
                "Class-wise routed logits require an nn.Linear classifier."
            )

        weight = self.classifier.weight
        bias = self.classifier.bias

        audio_logits = F.linear(audio_feature, weight, bias=None)
        guided_visual_logits = F.linear(
            guided_visual_feature, weight, bias=None
        )
        uniform_visual_logits = F.linear(
            uniform_visual_feature, weight, bias=None
        )

        gate = self.class_attention_gate[: weight.shape[0]].to(
            device=weight.device,
            dtype=audio_logits.dtype,
        )
        gate = gate.view(1, -1)

        logits = (
            audio_logits
            + gate * guided_visual_logits
            + (1.0 - gate) * uniform_visual_logits
        )

        if bias is not None:
            logits = logits + bias.view(1, -1)

        return logits

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
        out_uniform_visual=False,
    ):
        # The experiment is defined for the audio-visual model.  Preserve the
        # original implementation for any other modality.
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
            raise ValueError(
                "input frames are None when modality contains visual"
            )
        if audio is None:
            raise ValueError(
                "input audio is None when modality contains audio"
            )

        # Fixed visual tokens: (B, 8, P, 768).
        visual_4d = visual.reshape(visual.shape[0], 8, -1, 768)

        # Audio-independent uniform visual pooling.  No new parameters.
        visual_uniform_pooled = visual_4d.mean(dim=(1, 2))

        # Original audio-guided spatial + temporal visual pooling.
        spatial_attn_score, temporal_attn_score = (
            self.audio_visual_attention(audio, visual_4d)
        )
        visual_guided_pooled = torch.sum(
            spatial_attn_score * visual_4d, dim=2
        )
        visual_guided_pooled = torch.sum(
            temporal_attn_score * visual_guided_pooled, dim=1
        )

        audio_feature = F.relu(self.audio_proj(audio))
        guided_visual_feature = F.relu(
            self.visual_proj(visual_guided_pooled)
        )
        uniform_visual_feature = F.relu(
            self.visual_proj(visual_uniform_pooled)
        )

        # The original guided fused feature is retained for compatibility and
        # for the unchanged z1 contrastive losses.  Classification itself uses
        # class-wise routed logits below.
        guided_audio_visual_feature = (
            audio_feature + guided_visual_feature
        )

        logits = self._classwise_routed_logits(
            audio_feature=audio_feature,
            guided_visual_feature=guided_visual_feature,
            uniform_visual_feature=uniform_visual_feature,
        )

        if return_dict:
            outputs_dict = {}

            if AFC_train_out:
                audio_feature.retain_grad()
                guided_visual_feature.retain_grad()
                uniform_visual_feature.retain_grad()
                visual_guided_pooled.retain_grad()

                outputs_dict["logits"] = logits
                outputs_dict["visual_pooled_feature"] = (
                    visual_guided_pooled
                )
                outputs_dict["audio_feature"] = audio_feature
                outputs_dict["visual_feature"] = guided_visual_feature
                outputs_dict["uniform_visual_feature_raw"] = (
                    uniform_visual_feature
                )
                return outputs_dict

            if out_logits:
                outputs_dict["logits"] = logits

            if out_features:
                # There is no single feature whose one linear classification
                # reproduces class-wise routed logits.  Return the original
                # guided fused feature for backward-compatible diagnostics.
                if out_features_norm:
                    outputs_dict["features"] = F.normalize(
                        guided_audio_visual_feature, dim=1
                    )
                else:
                    outputs_dict["features"] = (
                        guided_audio_visual_feature
                    )

            if out_feature_before_fusion:
                # Keep the original Full AVCIL contrastive pathway unchanged.
                outputs_dict["audio_feature"] = F.normalize(
                    audio_feature, dim=1
                )
                outputs_dict["visual_feature"] = F.normalize(
                    guided_visual_feature, dim=1
                )

            if out_uniform_visual:
                outputs_dict["uniform_visual_feature_raw"] = (
                    uniform_visual_feature
                )
                outputs_dict["uniform_visual_feature_norm"] = F.normalize(
                    uniform_visual_feature, dim=1
                )

            if out_analysis_features:
                outputs_dict["z0_audio_raw"] = audio
                outputs_dict["z0_visual_uniform_raw"] = (
                    visual_uniform_pooled
                )
                outputs_dict["z0_audio_norm"] = F.normalize(
                    audio, dim=1
                )
                outputs_dict["z0_visual_uniform_norm"] = F.normalize(
                    visual_uniform_pooled, dim=1
                )

                outputs_dict["attn_visual_pooled_raw"] = (
                    visual_guided_pooled
                )
                outputs_dict["attn_visual_pooled_norm"] = F.normalize(
                    visual_guided_pooled, dim=1
                )

                outputs_dict["z1_audio_raw"] = audio_feature
                outputs_dict["z1_visual_raw"] = guided_visual_feature
                outputs_dict["z1_visual_uniform_raw"] = (
                    uniform_visual_feature
                )
                outputs_dict["z2_fusion_raw"] = (
                    guided_audio_visual_feature
                )

                outputs_dict["z1_audio_norm"] = F.normalize(
                    audio_feature, dim=1
                )
                outputs_dict["z1_visual_norm"] = F.normalize(
                    guided_visual_feature, dim=1
                )
                outputs_dict["z1_visual_uniform_norm"] = F.normalize(
                    uniform_visual_feature, dim=1
                )
                outputs_dict["z2_fusion_norm"] = F.normalize(
                    guided_audio_visual_feature, dim=1
                )
                outputs_dict["class_attention_gate"] = (
                    self.class_attention_gate[: self.classifier.out_features]
                )

            if out_z1_projection:
                if not getattr(
                    self, "use_z1_cm_projection_head", False
                ):
                    raise ValueError(
                        "out_z1_projection=True but "
                        "z1_cm_projection_head is not enabled"
                    )

                audio_z1_proj, visual_z1_proj = (
                    self.project_z1_features(
                        audio_feature,
                        guided_visual_feature,
                    )
                )
                outputs_dict["audio_z1_proj"] = audio_z1_proj
                outputs_dict["visual_z1_proj"] = visual_z1_proj

            if out_attn_score:
                outputs_dict["spatial_attn_score"] = spatial_attn_score
                outputs_dict["temporal_attn_score"] = (
                    temporal_attn_score
                )

            return outputs_dict

        outputs = ()

        if AFC_train_out:
            audio_feature.retain_grad()
            guided_visual_feature.retain_grad()
            visual_guided_pooled.retain_grad()
            outputs += (
                logits,
                visual_guided_pooled,
                audio_feature,
                guided_visual_feature,
            )
            return outputs

        if out_logits:
            outputs += (logits,)

        if out_features:
            if out_features_norm:
                outputs += (
                    F.normalize(guided_audio_visual_feature, dim=1),
                )
            else:
                outputs += (guided_audio_visual_feature,)

        if out_feature_before_fusion:
            outputs += (
                F.normalize(audio_feature, dim=1),
                F.normalize(guided_visual_feature, dim=1),
            )

        if out_z1_projection:
            if not getattr(self, "use_z1_cm_projection_head", False):
                raise ValueError(
                    "out_z1_projection=True but "
                    "z1_cm_projection_head is not enabled"
                )
            audio_z1_proj, visual_z1_proj = self.project_z1_features(
                audio_feature,
                guided_visual_feature,
            )
            outputs += (audio_z1_proj, visual_z1_proj)

        if out_attn_score:
            outputs += (spatial_attn_score, temporal_attn_score)

        if out_uniform_visual:
            outputs += (uniform_visual_feature,)

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def incremental_classifier(self, numclass: int) -> None:
        """Expand classifier and class gate while preserving learned rows."""
        if not isinstance(self.classifier, nn.Linear):
            raise TypeError(
                "Offline class-wise routing supports nn.Linear only."
            )

        old_classifier = self.classifier
        old_out_features = old_classifier.out_features
        device = old_classifier.weight.device
        dtype = old_classifier.weight.dtype

        new_classifier = nn.Linear(
            old_classifier.in_features,
            numclass,
            bias=old_classifier.bias is not None,
        ).to(device=device, dtype=dtype)

        with torch.no_grad():
            new_classifier.weight[:old_out_features].copy_(
                old_classifier.weight
            )
            if old_classifier.bias is not None:
                new_classifier.bias[:old_out_features].copy_(
                    old_classifier.bias
                )

        self.classifier = new_classifier
        self.num_classes = numclass
        self.class_attention_gate = self._make_class_gate(
            numclass,
            device=device,
            dtype=dtype,
        )