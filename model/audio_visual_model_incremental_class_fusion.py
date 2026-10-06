"""AVCIL branches with a detached, checkpointed audio gate for each class.

The default linear head and branch extraction match the phase-8 base model.
No target label is accepted by forward. MLP heads evaluate every candidate's
fused feature and assemble logits BEFORE the joint classification softmax.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .audio_visual_model_incremental import IncreAudioVisualNet

# P8 使用的共享模型来源：model/audio_visual_model_incremental.py::IncreAudioVisualNet。
# 下方标为“逻辑沿用”的部分对应该模型；P8 目录本身没有另一份模型实现。

# [P9 新增] 按候选类别融合并取该类别的 logit；线性/非线性头均无需真实标签。
def class_conditional_logits(audio, visual, classifier, gate, chunk_size=16):
    """z_c = classifier(2*g_c*a + 2*(1-g_c)*v)[c], without labels."""
    if isinstance(classifier, nn.Linear):
        # [P8 逻辑沿用] 这里保留原 forward 的 classifier(audio + visual) 基项；
        # 后面的差值修正为 P9 新增，g=0.5 时修正为零。
        # Algebraically identical to weighted branch logits. This form also
        # retains the original sum-fusion computation exactly when g == 0.5.
        return classifier(audio + visual) + F.linear(
            audio - visual, classifier.weight, bias=None
        ) * (2.0 * gate - 1.0)
    if chunk_size < 1:
        raise ValueError("Fusion chunk size must be positive")
    if torch.all(gate == 0.5):
        return classifier(audio + visual)
    result = []
    for start in range(0, gate.numel(), chunk_size):
        stop = min(start + chunk_size, gate.numel())
        weights = gate[start:stop].view(1, -1, 1)
        fused = 2.0 * weights * audio[:, None, :] + 2.0 * (1.0 - weights) * visual[:, None, :]
        scores = classifier(fused.reshape(-1, fused.shape[-1]))
        scores = scores.view(audio.shape[0], stop - start, gate.numel())
        indices = torch.arange(start, stop, device=audio.device)
        result.append(scores.gather(2, indices.view(1, -1, 1).expand(audio.shape[0], -1, 1)).squeeze(2))
    return torch.cat(result, dim=1)


class ClassFusionAudioVisualNet(IncreAudioVisualNet):
    def __init__(self, args, step_out_class_num):
        if args.modality != "audio-visual":
            raise ValueError("Class fusion requires audio-visual inputs")
        if getattr(args, "z1_cm_projection_head", False):
            raise ValueError("Phase 9 uses the original AVCIL branches, without a Z1 projector")
        # [P8 原样沿用] 直接调用共享基类初始化，继承原投影层、注意力层及默认线性头。
        super().__init__(args, step_out_class_num)
        # [P9 新增] 可选 MLP 分类器、类别融合权重及其版本缓冲区。
        self.fusion_classifier = getattr(args, "fusion_classifier", "linear")
        self.fusion_chunk_size = getattr(args, "fusion_chunk_size", 16)
        if self.fusion_classifier == "mlp":
            hidden = getattr(args, "fusion_hidden_dim", 768)
            self.classifier = nn.Sequential(nn.Linear(768, hidden), nn.ReLU(), nn.Linear(hidden, step_out_class_num))
        elif self.fusion_classifier != "linear":
            raise ValueError("fusion_classifier must be linear or mlp")
        self.register_buffer("fusion_gate", torch.full((step_out_class_num,), 0.5))
        self.register_buffer("fusion_version", torch.zeros((), dtype=torch.long))

    # [P8 逻辑沿用] 从基类 forward 的 audio-visual 分支抽出：空间池化 -> 时间池化
    # -> 两个投影层 + ReLU；变量名和 reshape 写法有调整，计算公式保留。
    # audio_visual_attention 方法直接继承基类；视觉分支仍含音频引导注意力。
    def extract_branch_features(self, visual, audio):
        if visual is None or audio is None:
            raise ValueError("Both visual and audio features are required")
        visual = visual.reshape(visual.shape[0], 8, -1, 768)
        spatial, temporal = self.audio_visual_attention(audio, visual)
        pooled = torch.sum(spatial * visual, dim=2)
        pooled = torch.sum(temporal * pooled, dim=1)
        audio_feature = F.relu(self.audio_proj(audio))
        visual_feature = F.relu(self.visual_proj(pooled))
        return audio_feature, visual_feature, spatial, temporal

    # [P8 逻辑沿用] 保留归一化分支特征、空间/时间注意力的输出顺序和单输出解包。
    # [P9 新增] logits 改由 class_conditional_logits 生成；不是原 forward 的完整复制。
    def forward(self, visual=None, audio=None, out_logits=True,
                out_feature_before_fusion=False, out_attn_score=False):
        a, v, spatial, temporal = self.extract_branch_features(visual, audio)
        outputs = ()
        if out_logits:
            outputs += (class_conditional_logits(a, v, self.classifier, self.fusion_gate, self.fusion_chunk_size),)
        if out_feature_before_fusion:
            outputs += (F.normalize(a, dim=1), F.normalize(v, dim=1))
        if out_attn_score:
            outputs += (spatial, temporal)
        return outputs[0] if len(outputs) == 1 else outputs

    # [P9 新增] 无梯度替换类别权重，供周期更新与 checkpoint 记录使用。
    @torch.no_grad()
    def set_fusion_gate(self, gate):
        gate = torch.as_tensor(gate, device=self.fusion_gate.device, dtype=self.fusion_gate.dtype)
        if gate.shape != self.fusion_gate.shape or not torch.isfinite(gate).all():
            raise ValueError("Gate must be finite and match the classifier's class order")
        if ((gate < 0) | (gate > 1)).any():
            raise ValueError("Gate must lie in [0, 1]")
        self.fusion_gate.copy_(gate.detach())
        self.fusion_version.add_(1)

    # [P8 逻辑沿用] 基类同名方法的“扩大输出层并保留旧类 weight/bias”策略。
    # [P9 新增] 兼容 MLP、保持 device/dtype，并扩展 gate：旧类继承，新类初始化为 0.5。
    @torch.no_grad()
    def incremental_classifier(self, numclass):
        old = self.classifier if self.fusion_classifier == "linear" else self.classifier[-1]
        if numclass <= old.out_features:
            raise ValueError("Incremental expansion must add classes")
        new = nn.Linear(old.in_features, numclass).to(device=old.weight.device, dtype=old.weight.dtype)
        new.weight[:old.out_features].copy_(old.weight)
        new.bias[:old.out_features].copy_(old.bias)
        if self.fusion_classifier == "linear":
            self.classifier = new
        else:
            self.classifier[-1] = new
        gate = self.fusion_gate.new_full((numclass,), 0.5)
        gate[:self.num_classes].copy_(self.fusion_gate)
        self.fusion_gate = gate
        self.num_classes = numclass

    # [P9 新增] 类别融合没有唯一的联合特征；阻止绕开类别融合的旧调用方式。
    def extract_joint_feature(self, visual=None, audio=None):
        raise ValueError("Class-conditional fusion has no single joint feature; use branch features or logits")

    def classify(self, features):
        raise ValueError("Use forward with both branches so class fusion cannot be bypassed")
