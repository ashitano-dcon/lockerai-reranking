import os
import logging
import torch
import hydra
from typing import Any

# 環境変数をスクリプト内で設定
os.environ["PROJECT_DIR"] = os.getcwd()
os.environ["HYDRA_FULL_ERROR"] = "1"

from fastapi import Depends, FastAPI, HTTPException
from omegaconf import OmegaConf
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import load_file as load_safetensor
from torch.cuda.amp import autocast

from .modeling import ModernBertForSequenceClassificationWithScalar

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# リクエストとレスポンスのモデル
class InferenceRequest(BaseModel):
    description: str
    inquiry: str
    latency: float | None = 0.0

class PredictionResponse(BaseModel):
    data: tuple[float, float]
    error: str | None = None

class ModelService:
    """モデルサービスクラス"""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.model: ModernBertForSequenceClassificationWithScalar | None = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_path: str | None = None) -> None:
        """モデルとトークナイザーをロードする。ローカルまたは Hugging Face Hub から取得可能"""
        if model_path is None:
            model_path = self.cfg.inference.model_path
        logger.info(f"モデルをロード中: {model_path}")

        try:
            # HF Hub or local
            if not os.path.isdir(model_path):
                token = os.getenv("HF_API_TOKEN", None)
                model = ModernBertForSequenceClassificationWithScalar.from_pretrained(
                    model_path,
                    revision=getattr(self.cfg.inference, "revision", None),
                    use_auth_token=token
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    revision=getattr(self.cfg.inference, "revision", None),
                    use_auth_token=token
                )
            else:
                config = AutoConfig.from_pretrained(model_path)
                model = ModernBertForSequenceClassificationWithScalar(config)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                # 重みファイル検出・ロード
                weight = None
                for fn in ("pytorch_model.safetensors", "pytorch_model.bin"):
                    p = os.path.join(model_path, fn)
                    if os.path.isfile(p): weight = p; break
                if weight is None:
                    raise FileNotFoundError(f"重みファイルが見つかりません: {model_path}")
                state = load_safetensor(weight) if weight.endswith(".safetensors") else torch.load(weight, map_location="cpu")
                model.load_state_dict(state)

            # Mixed precision for inference
            model.to(self.device)
            if self.device.type == 'cuda':
                model = model.half()
            model.eval()

            self.model = model
            self.tokenizer = tokenizer
            logger.info("モデルとトークナイザーのロード完了")
        except Exception as e:
            logger.error(f"モデルのロード中エラー: {e}")
            raise

    def predict(self, description: str, inquiry: str, latency: float = 0.0) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("モデルがロードされていません。load_model() を先に実行してください。")
        text = description + self.tokenizer.sep_token + inquiry
        enc = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        scalar = torch.tensor([[latency]], dtype=torch.float, device=self.device)

        with torch.no_grad(), autocast(self.device.type):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, scalar=scalar)
            probs = torch.nn.functional.softmax(out.logits, dim=1)[0]
        return {"data": (float(probs[0]), float(probs[1]))}

    def batch_predict(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """メモリを節約しつつチャンクごとに推論"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("モデルがロードされていません。load_model() を先に実行してください。")
        bs = getattr(self.cfg.inference, "batch_size", 4)
        results: list[dict[str, Any]] = []

        for i in range(0, len(items), bs):
            chunk = items[i:i+bs]
            texts = [c["description"] + self.tokenizer.sep_token + c["inquiry"] for c in chunk]
            lat = [float(c.get("latency", 0.0)) for c in chunk]
            enc = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            scalars = torch.tensor(lat, dtype=torch.float, device=self.device).view(-1,1)

            with torch.no_grad(), autocast(self.device.type):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, scalar=scalars)
                probs = torch.nn.functional.softmax(out.logits, dim=1)

            for idx in range(len(chunk)):
                results.append({"data": (float(probs[idx,0]), float(probs[idx,1]))})

            # メモリ解放
            torch.cuda.empty_cache()
        return results

# FastAPI setup
app = FastAPI(title="LockerAI Reranking API")
model_service: ModelService | None = None

def get_model_service() -> ModelService:
    if model_service is None:
        raise RuntimeError("モデルサービスが未初期化です。")
    return model_service

@app.post("/predict", response_model=PredictionResponse)
async def api_predict(req: InferenceRequest, svc: ModelService = Depends(get_model_service)):
    try:
        return PredictionResponse(**svc.predict(req.description, req.inquiry, req.latency or 0.0))
    except Exception as e:
        return PredictionResponse(data=(0.0,0.0), error=str(e))

@app.post("/batch_predict", response_model=list[PredictionResponse])
async def api_batch(reqs: list[InferenceRequest], svc: ModelService = Depends(get_model_service)):
    try:
        items = [r.model_dump() for r in reqs]
        out = svc.batch_predict(items)
        return [PredictionResponse(**r) for r in out]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: Any):
    global model_service
    logger.info(OmegaConf.to_yaml(cfg))
    model_service = ModelService(cfg)
    model_service.load_model()
    import uvicorn
    uvicorn.run(app, host=cfg.inference.host, port=cfg.inference.port)

if __name__ == "__main__":
    main()
