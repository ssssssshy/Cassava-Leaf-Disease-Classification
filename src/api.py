import os
import torch
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from typing import Dict, Any
from contextlib import asynccontextmanager
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2

from src.models import build_model
from src.config import load_config


class FastAPIPredictor:
    def __init__(self, config_path: str):
        self.cfg = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = self.cfg.train.img_size

        self.onnx_path = Path(self.cfg.path.save_path).with_suffix(".onnx")
        self.pth_path = Path(self.cfg.path.save_path)

        if self.onnx_path.exists():
            print(f"Loading ONNX model from {self.onnx_path}")
            self.mode = "onnx"
            self.session = ort.InferenceSession(str(self.onnx_path))
            self.input_name = self.session.get_inputs()[0].name
        elif self.pth_path.exists():
            print(f"Loading PyTorch model from {self.pth_path}")
            self.mode = "torch"
            self.model, _ = build_model(self.cfg)
            checkpoint = torch.load(self.pth_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
        else:
            raise FileNotFoundError(
                f"Neither {self.pth_path} nor {self.onnx_path} exists."
            )

        self.transform = v2.Compose(
            [
                v2.Resize((self.img_size, self.img_size)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
            ]
        )

        self.class_names = {
            0: "Cassava Bacterial Blight (CBB)",
            1: "Cassava Brown Streak Disease (CBSD)",
            2: "Cassava Green Mottle (CGM)",
            3: "Cassava Mosaic Disease (CMD)",
            4: "Healthy",
        }

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        tensor = decode_image(
            torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8),
            mode=ImageReadMode.RGB,
        )

        processed_tensor = self.transform(tensor)
        return processed_tensor.unsqueeze(0).numpy()

    @torch.inference_mode()
    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        input_tensor = self.preprocess(image_bytes)

        if self.mode == "onnx":
            outputs = self.session.run(None, {self.input_name: input_tensor})[0]
            probs = self._softmax(outputs)[0]  # type: ignore
        else:
            input_torch = torch.from_numpy(input_tensor).to(self.device)
            outputs = self.model(input_torch)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])

        probabilities_dict = {str(i): float(prob) for i, prob in enumerate(probs)}

        return {
            "prediction": {
                "top_class_id": top_idx,
                "top_class_name": self.class_names.get(top_idx, "Unknown"),
                "confidence": confidence,
            },
            "probabilities": probabilities_dict,
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)


predictor: FastAPIPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global predictor
    config_path = os.getenv("MODEL_CONFIG", "configs/yolo12.yaml")
    predictor = FastAPIPredictor(config_path)

    yield

    predictor = None


app = FastAPI(title="Imbalance CV Pipeline API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "mode": predictor.mode if predictor else "initializing"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model is not initialized yet")

    try:
        content = await file.read()
        result = predictor.predict(content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Imbalance CV Pipeline API is running"}
