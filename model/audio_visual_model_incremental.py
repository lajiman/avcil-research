import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import LSCLinear, SplitLSCLinear


class IncreAudioVisualNet(nn.Module):
    def __init__(self, args, step_out_class_num, LSC=False):
        super(IncreAudioVisualNet, self).__init__()

        self.args = args
        self.modality = args.modality
        self.num_classes = step_out_class_num

        if self.modality != 'visual' and self.modality != 'audio' and self.modality != 'audio-visual':
            raise ValueError('modality must be \'visual\', \'audio\' or \'audio-visual\'')

        if self.modality == 'visual':
            self.visual_proj = nn.Linear(768, 768)

        elif self.modality == 'audio':
            self.audio_proj = nn.Linear(768, 768)

        else:
            self.audio_proj = nn.Linear(768, 768)
            self.visual_proj = nn.Linear(768, 768)
            self.attn_audio_proj = nn.Linear(768, 768)
            self.attn_visual_proj = nn.Linear(768, 768)

            # ---------------------------------------------------------
            # Z1 projection head for cross-modal contrastive loss.
            #
            # Important:
            #   - classifier still uses original audio_feature + visual_feature
            #   - contrastive loss may use projected audio_z1_proj / visual_z1_proj
            # ---------------------------------------------------------
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

        if LSC:
            self.classifier = LSCLinear(768, self.num_classes)
        else:
            self.classifier = nn.Linear(768, self.num_classes)

    def _build_z1_projection_head(self, in_dim, hidden_dim, out_dim, head_type="mlp"):
        """
        Projection head used only for z1 cross-modal contrastive loss.
        """
        if head_type == "linear":
            return nn.Linear(in_dim, out_dim)

        elif head_type == "mlp":
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )

        else:
            raise ValueError("z1_cm_projection_type must be 'linear' or 'mlp'")

    def project_z1_features(self, audio_feature, visual_feature):
        """
        Project z1 pre-fusion audio/visual features into contrastive space.

        audio_feature:
            Raw z1 audio feature after audio_proj + ReLU, shape=(B, 768)

        visual_feature:
            Raw z1 visual feature after visual_proj + ReLU, shape=(B, 768)

        returns:
            audio_z:
                Normalized projected audio feature.

            visual_z:
                Normalized projected visual feature.
        """
        if not getattr(self, "use_z1_cm_projection_head", False):
            raise ValueError("z1_cm_projection_head is not enabled in this model")

        audio_z = self.z1_audio_projector(audio_feature)
        visual_z = self.z1_visual_projector(visual_feature)

        audio_z = F.normalize(audio_z, dim=1)
        visual_z = F.normalize(visual_z, dim=1)

        return audio_z, visual_z

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
        if self.modality == 'visual':
            if visual is None:
                raise ValueError('input frames are None when modality contains visual')

            visual_feature = torch.mean(visual, dim=1)
            visual_feature = F.relu(self.visual_proj(visual_feature))
            logits = self.classifier(visual_feature)

            # ---------------------------------------------------------
            # Dict output branch
            # ---------------------------------------------------------
            if return_dict:
                outputs_dict = {}

                if AFC_train_out:
                    visual_feature.retain_grad()
                    outputs_dict["logits"] = logits
                    outputs_dict["visual_feature"] = visual_feature
                    return outputs_dict

                if out_logits:
                    outputs_dict["logits"] = logits

                if out_features:
                    if out_features_norm:
                        outputs_dict["features"] = F.normalize(visual_feature, dim=1)
                    else:
                        outputs_dict["features"] = visual_feature

                return outputs_dict

            # ---------------------------------------------------------
            # Original tuple/tensor output branch
            # ---------------------------------------------------------
            outputs = ()

            if AFC_train_out:
                visual_feature.retain_grad()
                outputs += (logits, visual_feature)
                return outputs

            else:
                if out_logits:
                    outputs += (logits,)

                if out_features:
                    if out_features_norm:
                        outputs += (F.normalize(visual_feature, dim=1),)
                    else:
                        outputs += (visual_feature,)

                if len(outputs) == 1:
                    return outputs[0]
                else:
                    return outputs

        elif self.modality == 'audio':
            if audio is None:
                raise ValueError('input audio are None when modality contains audio')

            audio_feature = F.relu(self.audio_proj(audio))
            logits = self.classifier(audio_feature)

            # ---------------------------------------------------------
            # Dict output branch
            # ---------------------------------------------------------
            if return_dict:
                outputs_dict = {}

                if AFC_train_out:
                    audio_feature.retain_grad()
                    outputs_dict["logits"] = logits
                    outputs_dict["audio_feature"] = audio_feature
                    return outputs_dict

                if out_logits:
                    outputs_dict["logits"] = logits

                if out_features:
                    if out_features_norm:
                        outputs_dict["features"] = F.normalize(audio_feature, dim=1)
                    else:
                        outputs_dict["features"] = audio_feature

                return outputs_dict

            # ---------------------------------------------------------
            # Original tuple/tensor output branch
            # ---------------------------------------------------------
            outputs = ()

            if AFC_train_out:
                audio_feature.retain_grad()
                outputs += (logits, audio_feature)
                return outputs

            else:
                if out_logits:
                    outputs += (logits,)

                if out_features:
                    if out_features_norm:
                        outputs += (F.normalize(audio_feature, dim=1),)
                    else:
                        outputs += (audio_feature,)

                if len(outputs) == 1:
                    return outputs[0]
                else:
                    return outputs

        else:
            if visual is None:
                raise ValueError('input frames are None when modality contains visual')

            if audio is None:
                raise ValueError('input audio are None when modality contains audio')

            # Keep an explicit 4-D view so analysis can expose a fixed,
            # audio-independent visual z0 representation without altering training.
            visual_4d = visual.view(visual.shape[0], 8, -1, 768)
            visual_z0_uniform = visual_4d.mean(dim=(1, 2))

            spatial_attn_score, temporal_attn_score = self.audio_visual_attention(audio, visual_4d)

            visual_pooled_feature = torch.sum(spatial_attn_score * visual_4d, dim=2)
            visual_pooled_feature = torch.sum(temporal_attn_score * visual_pooled_feature, dim=1)

            audio_feature = F.relu(self.audio_proj(audio))
            visual_feature = F.relu(self.visual_proj(visual_pooled_feature))

            audio_visual_features = visual_feature + audio_feature
            logits = self.classifier(audio_visual_features)

            # ---------------------------------------------------------
            # Dict output branch
            # ---------------------------------------------------------
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
                    if out_features_norm:
                        outputs_dict["features"] = F.normalize(audio_visual_features, dim=1)
                    else:
                        outputs_dict["features"] = audio_visual_features

                if out_feature_before_fusion:
                    # Preserve the original public behavior used by existing training code.
                    outputs_dict["audio_feature"] = F.normalize(audio_feature, dim=1)
                    outputs_dict["visual_feature"] = F.normalize(visual_feature, dim=1)

                if out_analysis_features:
                    # Fixed model inputs (z0). visual_z0_uniform is independent of
                    # audio-guided attention and is therefore suitable as a static
                    # visual-reference embedding.
                    outputs_dict["z0_audio_raw"] = audio
                    outputs_dict["z0_visual_uniform_raw"] = visual_z0_uniform
                    outputs_dict["z0_audio_norm"] = F.normalize(audio, dim=1)
                    outputs_dict["z0_visual_uniform_norm"] = F.normalize(visual_z0_uniform, dim=1)

                    # Audio-conditioned visual representation before visual_proj.
                    outputs_dict["attn_visual_pooled_raw"] = visual_pooled_feature
                    outputs_dict["attn_visual_pooled_norm"] = F.normalize(
                        visual_pooled_feature, dim=1
                    )

                    # Raw representations actually consumed by the additive fusion
                    # and the linear classifier. These must be used for counterfactual
                    # branch interventions; normalized vectors would change the model.
                    outputs_dict["z1_audio_raw"] = audio_feature
                    outputs_dict["z1_visual_raw"] = visual_feature
                    outputs_dict["z2_fusion_raw"] = audio_visual_features

                    # Explicit normalized copies for cosine geometry.
                    outputs_dict["z1_audio_norm"] = F.normalize(audio_feature, dim=1)
                    outputs_dict["z1_visual_norm"] = F.normalize(visual_feature, dim=1)
                    outputs_dict["z2_fusion_norm"] = F.normalize(
                        audio_visual_features, dim=1
                    )

                if out_z1_projection:
                    if not getattr(self, "use_z1_cm_projection_head", False):
                        raise ValueError(
                            "out_z1_projection=True but z1_cm_projection_head is not enabled"
                        )

                    audio_z1_proj, visual_z1_proj = self.project_z1_features(
                        audio_feature,
                        visual_feature,
                    )

                    outputs_dict["audio_z1_proj"] = audio_z1_proj
                    outputs_dict["visual_z1_proj"] = visual_z1_proj

                if out_attn_score:
                    outputs_dict["spatial_attn_score"] = spatial_attn_score
                    outputs_dict["temporal_attn_score"] = temporal_attn_score

                return outputs_dict

            # ---------------------------------------------------------
            # Original tuple/tensor output branch
            # ---------------------------------------------------------
            outputs = ()

            if AFC_train_out:
                audio_feature.retain_grad()
                visual_feature.retain_grad()
                visual_pooled_feature.retain_grad()
                outputs += (logits, visual_pooled_feature, audio_feature, visual_feature)
                return outputs

            else:
                if out_logits:
                    outputs += (logits,)

                if out_features:
                    if out_features_norm:
                        outputs += (F.normalize(audio_visual_features, dim=1),)
                    else:
                        outputs += (audio_visual_features,)

                if out_feature_before_fusion:
                    outputs += (F.normalize(audio_feature, dim=1), F.normalize(visual_feature, dim=1))

                if out_z1_projection:
                    if not getattr(self, "use_z1_cm_projection_head", False):
                        raise ValueError(
                            "out_z1_projection=True but z1_cm_projection_head is not enabled"
                        )

                    audio_z1_proj, visual_z1_proj = self.project_z1_features(audio_feature, visual_feature)

                    outputs += (audio_z1_proj, visual_z1_proj)

                if out_attn_score:
                    outputs += (spatial_attn_score, temporal_attn_score)

                if len(outputs) == 1:
                    return outputs[0]
                else:
                    return outputs

    def audio_visual_attention(self, audio_features, visual_features):
        proj_audio_features = torch.tanh(self.attn_audio_proj(audio_features))
        proj_visual_features = torch.tanh(self.attn_visual_proj(visual_features))

        # visual_features: (BS, 8, 14*14, 768)
        # audio_features:  (BS, 768)
        spatial_score = torch.einsum("ijkd,id->ijkd", [proj_visual_features, proj_audio_features])
        # (BS, 8, 14*14, 768)
        spatial_attn_score = F.softmax(spatial_score, dim=2)
        # (BS, 8, 768)
        spatial_attned_proj_visual_features = torch.sum(spatial_attn_score * proj_visual_features, dim=2)

        # (BS, 8, 768)
        temporal_score = torch.einsum("ijd,id->ijd", [spatial_attned_proj_visual_features, proj_audio_features])
        temporal_attn_score = F.softmax(temporal_score, dim=1)

        return spatial_attn_score, temporal_attn_score

    def incremental_classifier(self, numclass):
        old_classifier = self.classifier
        out_features = old_classifier.out_features

        # Keep the CPU initialization used by previous runs, then move the new
        # head to the existing model's device and dtype (including CUDA/BF16).
        new_classifier = nn.Linear(old_classifier.in_features, numclass, bias=True)
        new_classifier = new_classifier.to(
            device=old_classifier.weight.device, dtype=old_classifier.weight.dtype
        )
        with torch.no_grad():
            new_classifier.weight[:out_features].copy_(old_classifier.weight)
            new_classifier.bias[:out_features].copy_(old_classifier.bias)

        self.classifier = new_classifier
        self.num_classes = numclass

    def extract_joint_feature(self, visual=None, audio=None):
        """
        Return fused joint feature, shape=(B, 768).
        Used by stage2: freeze feature extractor, train classifier / LWS.
        """
        if self.modality == 'visual':
            if visual is None:
                raise ValueError('input frames are None when modality contains visual')

            visual_feature = torch.mean(visual, dim=1)
            visual_feature = F.relu(self.visual_proj(visual_feature))
            return visual_feature

        elif self.modality == 'audio':
            if audio is None:
                raise ValueError('input audio are None when modality contains audio')

            audio_feature = F.relu(self.audio_proj(audio))
            return audio_feature

        else:
            if visual is None:
                raise ValueError('input frames are None when modality contains visual')

            if audio is None:
                raise ValueError('input audio are None when modality contains audio')

            visual = visual.view(visual.shape[0], 8, -1, 768)

            spatial_attn_score, temporal_attn_score = self.audio_visual_attention(audio, visual)

            visual_pooled_feature = torch.sum(spatial_attn_score * visual, dim=2)
            visual_pooled_feature = torch.sum(temporal_attn_score * visual_pooled_feature, dim=1)

            audio_feature = F.relu(self.audio_proj(audio))
            visual_feature = F.relu(self.visual_proj(visual_pooled_feature))

            audio_visual_features = visual_feature + audio_feature

            return audio_visual_features

    def classify(self, features):
        return self.classifier(features)
