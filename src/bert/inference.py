import os
import logging
import torch
import hydra
from typing import Any

# ngrok 接続用
from pyngrok import ngrok

# ngrok auth token を環境変数から設定
auth_token = os.getenv("NGROKAUTH_TOKEN")
if auth_token:
    ngrok.set_auth_token(auth_token)
else:
    logger.warning("環境変数 NGROK_AUTH_TOKEN が設定されていません。無料トンネルはすぐに切断される可能性があります。")

from fastapi import Depends, FastAPI, HTTPException
from omegaconf import OmegaConf
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import load_file as load_safetensor

from .modeling import ModernBertForSequenceClassificationWithScalar

# 環境変数設定
os.environ["PROJECT_DIR"] = os.getcwd()
os.environ["HYDRA_FULL_ERROR"] = "1"

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
        """モデルとトークナイザーをロードする。ローカルまたは Hub から取得"""
        if model_path is None:
            model_path = self.cfg.inference.model_path
        logger.info(f"モデルをロード中: {model_path}")
        try:
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
                logger.info("Hub からロード完了")
                return
            config = AutoConfig.from_pretrained(model_path)
            model = ModernBertForSequenceClassificationWithScalar(config)
            weight_file = None
            for fname in ("pytorch_model.safetensors","pytorch_model.bin"):
                p = os.path.join(model_path, fname)
                if os.path.isfile(p): weight_file = p; break
            if not weight_file:
                raise FileNotFoundError(f"重みが見つからず: {model_path}")
            state = load_safetensor(weight_file) if weight_file.endswith(".safetensors") else torch.load(weight_file, map_location="cpu")
            model.load_state_dict(state)
            model.to(self.device).eval()
            self.model = model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("ローカルからロード完了")
        except Exception as e:
            logger.error(f"ロード失敗: {e}")
            raise

    def predict(self, description: str, inquiry: str, latency: float = 0.0) -> dict[str,Any]:
        if not self.model or not self.tokenizer:
            raise RuntimeError("モデル未ロード")
        text = description + self.tokenizer.sep_token + inquiry
        enc = self.tokenizer(text, truncation=True,
                             max_length=self.model.config.max_position_embeddings,
                             padding="max_length", return_tensors="pt")
        scalar = torch.tensor([[latency]], dtype=torch.float).to(self.device)
        input_ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        with torch.no_grad(): out = self.model(input_ids=input_ids,
                                               attention_mask=mask,
                                               scalar=scalar)
        probs = torch.nn.functional.softmax(out.logits, dim=1)[0].tolist()
        return {"data": tuple(probs)}

    def batch_predict(self, items: list[dict[str,Any]]) -> list[dict[str,Any]]:
        if not self.model or not self.tokenizer:
            raise RuntimeError("モデル未ロード")
        texts = [i["description"]+self.tokenizer.sep_token+i["inquiry"] for i in items]
        lats  = [float(i.get("latency",0.0)) for i in items]
        encs  = self.tokenizer(texts, truncation=True,
                               max_length=self.model.config.max_position_embeddings,
                               padding="max_length", return_tensors="pt")
        scalars = torch.tensor(lats, dtype=torch.float).view(len(items),1).to(self.device)
        ids = encs["input_ids"].to(self.device)
        masks = encs["attention_mask"].to(self.device)
        with torch.no_grad(): out = self.model(input_ids=ids,
                                              attention_mask=masks,
                                              scalar=scalars)
        ps = torch.nn.functional.softmax(out.logits, dim=1)
        return [{"data": tuple(ps[i].tolist())} for i in range(len(items))]

app = FastAPI(title="LockerAI Reranking API")
model_service: ModelService|None = None

def get_model():
    if not model_service: raise RuntimeError("Service 未初期化")
    return model_service

@app.post("/predict", response_model=PredictionResponse)
async def api_predict(req:InferenceRequest, svc:ModelService=Depends(get_model)):
    try: return PredictionResponse(**svc.predict(req.description,req.inquiry,req.latency or 0.0))
    except Exception as e: return PredictionResponse(data=(0.0,0.0), error=str(e))

@app.post("/batch_predict", response_model=list[PredictionResponse])
async def api_batch(req:list[InferenceRequest], svc:ModelService=Depends(get_model)):
    try:
        items=[r.model_dump() for r in req]
        res=svc.batch_predict(items)
        return [PredictionResponse(**r) for r in res]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg:Any):
    global model_service
    logger.info(OmegaConf.to_yaml(cfg))
    model_service=ModelService(cfg)
    model_service.load_model()
    # ngrok トンネル開始
    public_url = ngrok.connect(cfg.inference.port).public_url
    logger.info(f"ngrok URL: {public_url}")
    uvicorn.run(app, host="0.0.0.0", port=cfg.inference.port)

if __name__=="__main__": main()
