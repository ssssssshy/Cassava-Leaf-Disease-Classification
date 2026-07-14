import argparse
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

from src.models import build_model
from src.config import load_config


def export_to_onnx(config_path: str, output_path: str | None = None) -> None:
    cfg = load_config(config_path)
    model, data_config = build_model(cfg)

    checkpoint_path = Path(cfg.path.save_path)
    if checkpoint_path.exists():
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(
            f"Warning: No checkpoint found at {checkpoint_path}. Exporting base model."
        )

    model.eval()

    img_size = cfg.train.img_size
    dummy_input = torch.randn(1, 3, img_size, img_size)

    if output_path is None:
        final_output_path = checkpoint_path.with_suffix(".onnx")
    else:
        final_output_path = Path(output_path)

    final_output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting model to {final_output_path}...")

    torch.onnx.export(
        model,
        (dummy_input,),
        str(final_output_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print("Checking ONNX model...")
    onnx_model = onnx.load(str(final_output_path))
    onnx.checker.check_model(onnx_model)

    print("Validating with ONNX Runtime...")
    ort_session = ort.InferenceSession(str(final_output_path))

    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    with torch.no_grad():
        torch_out = model(dummy_input)
        if isinstance(torch_out, (tuple, list)):
            torch_out = torch_out[0]

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(
        to_numpy(torch_out), np.array(ort_outs[0]), rtol=1e-03, atol=1e-05
    )
    print("Export successful! PyTorch and ONNX outputs match.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    export_to_onnx(args.config, args.output)
