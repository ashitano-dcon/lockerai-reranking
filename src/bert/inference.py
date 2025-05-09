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
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path: str | None = None) -> None:
        """モデルとトークナイザーをロードする。ローカルディレクトリまたは Hugging Face Hub から取得可能"""
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
                model.to(self.device).eval()
                self.model = model
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    revision=getattr(self.cfg.inference, "revision", None),
                    use_auth_token=token
                )
                logger.info("Hugging Face Hub からモデルとトークナイザーをロードしました")
                return

            # local dir load
            config = AutoConfig.from_pretrained(model_path)
            model = ModernBertForSequenceClassificationWithScalar(config)

            # find weight file
            weight_file = None
            for fname in ("pytorch_model.safetensors", "pytorch_model.bin"):
                candidate = os.path.join(model_path, fname)
                if os.path.isfile(candidate):
                    weight_file = candidate
                    break
            if weight_file is None:
                raise FileNotFoundError(f"重みファイルが見つかりません: {model_path}")

            if weight_file.endswith(".safetensors"):
                state = load_safetensor(weight_file)
            else:
                state = torch.load(weight_file, map_location="cpu")
            model.load_state_dict(state)

            model.to(self.device).eval()
            self.model = model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("ローカルからモデルとトークナイザーのロードが完了しました")
        except Exception as e:
            logger.error(f"モデルのロード中にエラーが発生しました: {e}")
            raise

    def predict(self, description: str, inquiry: str, latency: float = 0.0) -> dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("モデルとトークナイザーがロードされていません。load_model() を先に実行してください。")
        text = description + self.tokenizer.sep_token + inquiry
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
            padding="max_length",
            return_tensors="pt"
        )
        scalar = torch.tensor([[latency]], dtype=torch.float).to(self.device)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                scalar=scalar
            )
        probs = torch.nn.functional.softmax(out.logits, dim=1)[0].tolist()
        return {"data": tuple(probs)}

    def batch_predict(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """バッチをチャンクに分割してメモリ不足を回避しながら推論を行う"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("モデルとトークナイザーがロードされていません。load_model() を先に実行してください。")
        # チャンクサイズは設定またはデフォルト8
        batch_size = getattr(self.cfg.inference, "batch_size", 8)
        results: list[dict[str, Any]] = []

        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            texts = [itm["description"] + self.tokenizer.sep_token + itm["inquiry"] for itm in chunk]
            latencies = [float(itm.get("latency", 0.0)) for itm in chunk]

            encs = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.model.config.max_position_embeddings,
                padding="max_length",
                return_tensors="pt"
            )
            scalars = torch.tensor(latencies, dtype=torch.float).view(len(chunk), 1).to(self.device)
            input_ids = encs["input_ids"].to(self.device)
            attention_mask = encs["attention_mask"].to(self.device)

            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    scalar=scalars
                )
            probs = torch.nn.functional.softmax(out.logits, dim=1)

            for idx in range(len(chunk)):
                results.append({"data": tuple(probs[idx].tolist())})

        return results

# FastAPI setup
app = FastAPI(title="LockerAI Reranking API", description="ModernBERT based reranking API")
model_service: ModelService | None = None

def get_model_service() -> ModelService:
    if model_service is None:
        raise RuntimeError("モデルサービスが初期化されていません。サーバーを正しく起動してください。")
    return model_service

@app.post("/predict", response_model=PredictionResponse)
async def api_predict(request: InferenceRequest, service: ModelService = Depends(get_model_service)) -> PredictionResponse:
    try:
        res = service.predict(request.description, request.inquiry, request.latency or 0.0)
        return PredictionResponse(**res)
    except Exception as e:
        logger.error(f"推論中にエラーが発生しました: {e}")
        return PredictionResponse(data=(0.0,0.0), error=str(e))

@app.post("/batch_predict", response_model=list[PredictionResponse])
async def api_batch_predict(requests: list[InferenceRequest], service: ModelService = Depends(get_model_service)) -> list[PredictionResponse]:
    try:
        items = [r.model_dump() for r in requests]
        results = service.batch_predict(items)
        return [PredictionResponse(**r) for r in results]
    except Exception as e:
        logger.error(f"バッチ推論中にエラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: Any) -> None:
    global model_service
    logger.info(OmegaConf.to_yaml(cfg))
    model_service = ModelService(cfg)
    model_service.load_model()
    import uvicorn
    uvicorn.run(app, host=cfg.inference.host, port=cfg.inference.port)

if __name__ == "__main__":
    main()
