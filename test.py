"""Synthetic H200 smoke test; no dataset or existing checkpoint is needed.

Run ``python test.py --require-h200`` on the server, or explicitly select
``--device cuda:1`` / ``--device cpu``. This is not a full training validation.
"""

import argparse
import copy
import io
import platform
import sys
from types import SimpleNamespace


def run(args):
    import torch
    from torch.nn import functional as F

    from model.audio_visual_model_incremental import IncreAudioVisualNet
    from experiments_phase_8_rdcrosssdc_modular_gridsearch.rd_crosssdc.exact_losses import (
        ce_loss,
        cross_sdc_z1_loss,
    )
    from experiments_phase_8_rdcrosssdc_modular_gridsearch.rd_crosssdc.rd_method import (
        TeacherPrototypeBank,
        cmr_loss,
        compute_margin_terms,
    )

    def require(condition, message):
        if not condition:
            raise RuntimeError(message)

    def finite(tensor, name):
        require(torch.isfinite(tensor).all().item(), f"Non-finite {name}")

    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}; wheel CUDA runtime: {torch.version.cuda}")
    device = torch.device(args.device)
    require(device.type in {"cpu", "cuda"}, "Use --device cpu or cuda[:index]")
    bf16_supported = False
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is disabled")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
        props = torch.cuda.get_device_properties(device)
        bf16_supported = torch.cuda.is_bf16_supported()
        print(f"GPU: {props.name}; device: {device}; memory: {props.total_memory / 2**30:.1f} GiB")
        print(f"Compute capability: {props.major}.{props.minor}; BF16: {bf16_supported}")
        print(f"Wheel CUDA architectures: {torch.cuda.get_arch_list()}")
        if args.require_h200:
            require("H200" in props.name.upper(), f"Expected H200, found {props.name}")
    else:
        require(not args.require_h200, "--require-h200 requires a CUDA device")
        print("Device: CPU (explicit selection); GPU/compute capability/BF16: not tested")

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        # Keep the smoke-test training path in FP32, without TF32 or autocast.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    model = IncreAudioVisualNet(
        SimpleNamespace(modality="audio-visual", z1_cm_projection_head=False),
        step_out_class_num=4,
    ).to(device)

    # A non-default dtype catches a head replacement that silently resets dtype.
    model.classifier.double()
    old_weight = model.classifier.weight.detach().clone()
    old_bias = model.classifier.bias.detach().clone()
    model.incremental_classifier(6)
    require(model.classifier.out_features == 6, "Classifier did not grow")
    for name, old in (("weight", old_weight), ("bias", old_bias)):
        current = getattr(model.classifier, name)
        require(current.device == old.device, f"Classifier {name} changed device")
        require(current.dtype == old.dtype, f"Classifier {name} changed dtype")
        require(torch.equal(current[:4], old), f"Old classifier {name} changed")
    model.float()
    print("PASS: classifier expansion preserves old weights, biases, device and dtype")

    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    labels = torch.tensor([0, 0, 1, 1], device=device)
    audio = torch.randn(4, 768, device=device)
    # Eight frames are mandatory; four spatial tokens keep this test small.
    visual = torch.randn(4, 8, 4, 768, device=device)
    forward_options = {"out_feature_before_fusion": True, "return_dict": True}
    with torch.no_grad():
        old = teacher(audio=audio, visual=visual, **forward_options)
    current = model(
        audio=audio + 0.05 * torch.randn_like(audio),
        visual=visual + 0.05 * torch.randn_like(visual),
        **forward_options,
    )
    require(current["logits"].shape == (4, 6), "Unexpected expanded logits shape")
    for name, tensor in current.items():
        finite(tensor, name)
        require(tensor.dtype == torch.float32, f"Expected FP32 output: {name}")

    loss_i, loss_c = cross_sdc_z1_loss(
        current["audio_feature"], current["visual_feature"],
        old["audio_feature"], old["visual_feature"], labels,
    )
    # Two old classes, with two replay examples each, exercise leave-one-out CMR.
    audio_sums = torch.zeros(2, 768, device=device).index_add_(0, labels, old["audio_feature"])
    visual_sums = torch.zeros_like(audio_sums).index_add_(0, labels, old["visual_feature"])
    bank = TeacherPrototypeBank(
        audio_sums=audio_sums,
        visual_sums=visual_sums,
        audio_prototypes=F.normalize(audio_sums, dim=1),
        visual_prototypes=F.normalize(visual_sums, dim=1),
        counts=torch.full((2,), 2.0, device=device),
    )
    terms = compute_margin_terms(
        current["audio_feature"], current["visual_feature"],
        old["audio_feature"], old["visual_feature"], labels, bank,
        temperature=0.05, tolerance=0.0,
    )
    weights = torch.ones(2, device=device)
    loss_cmr, _ = cmr_loss(terms, labels, weights, weights)
    losses = {
        "CE": ce_loss(6, current["logits"], labels),
        "CrossSDC-I": loss_i, "CrossSDC-C": loss_c, "CMR": loss_cmr,
    }
    for name, loss in losses.items():
        finite(loss, name)
    total = sum(losses.values())
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    for name, param in model.named_parameters():
        require(param.grad is not None, f"Missing gradient: {name}")
        finite(param.grad, f"gradient {name}")
    before_step = model.classifier.weight.detach().clone()
    optimizer.step()
    require(not torch.equal(before_step, model.classifier.weight), "Adam did not update weights")
    for name, param in model.named_parameters():
        finite(param, f"updated parameter {name}")
    print("PASS: FP32 forward, real losses, finite gradients and Adam step")
    print("Losses: " + ", ".join(f"{name}={loss.item():.6f}" for name, loss in losses.items()))

    model.eval()
    with torch.no_grad():
        expected = model(audio=audio, visual=visual)
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    # This object was just created here; never use False for untrusted checkpoints.
    restored = torch.load(buffer, map_location=device, weights_only=False).eval()
    with torch.no_grad():
        actual = restored(audio=audio, visual=visual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    require(all(p.device == device for p in restored.parameters()), "Checkpoint device mismatch")
    print("PASS: full-model checkpoint round trip with explicit weights_only=False")

    if bf16_supported:
        matrix = torch.randn(128, 128, device=device, dtype=torch.bfloat16)
        finite(matrix @ matrix.T, "BF16 matrix multiplication")
        print("PASS: BF16 matrix multiplication (capability check only)")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        print(f"Peak allocated GPU memory: {torch.cuda.max_memory_allocated(device) / 2**20:.1f} MiB")
    print("PASS: synthetic smoke test; dataset I/O, full training and accuracy are not validated.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="cuda:0 (default), cuda:1, or explicit cpu")
    parser.add_argument("--require-h200", action="store_true", help="Fail unless the selected GPU is an H200")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
