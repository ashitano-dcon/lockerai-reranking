import os
import logging
import torch
import hydra
from typing import Any, List, Dict

# 環境変数をスクリプト内で設定
os.environ["PROJECT_DIR"] = os.getcwd()
os.environ["HYDRA_FULL_ERROR"] = "1"

from fastapi import Depends, FastAPI, HTTPException
from omegaconf import OmegaConf
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import load_file as load_safetensor
from torch.amp import autocast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.nn import CrossEntropyLoss

from .modeling import ModernBertForSequenceClassificationWithScalar

# ロガー設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# リクエスト／レスポンスモデル
class InferenceRequest(BaseModel):
    description: str
    inquiry: str
    latency: float | None = 0.0

class PredictionResponse(BaseModel):
    data: tuple[float, float]
    error: str | None = None

class EvalMetrics(BaseModel):
    eval_loss: float
    eval_precision: float
    eval_recall: float
    eval_f1: float
    eval_accuracy: float
    eval_runtime: float
    eval_samples_per_second: float
    eval_steps_per_second: float
    epoch: float

class ModelService:
    """モデルサービスクラス"""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.model: ModernBertForSequenceClassificationWithScalar | None = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = CrossEntropyLoss()

    def load_model(self, model_path: str | None = None) -> None:
        """モデルとトークナイザーをロードする。ローカルまたは Hugging Face Hub から取得可能"""
        if model_path is None:
            model_path = self.cfg.inference.model_path
        logger.info(f"モデルをロード中: {model_path}")

        try:
            # Hugging Face Hubまたはローカルディレクトリからロード
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
                # 重みファイル検出とロード
                weight_file = None
                for fn in ("pytorch_model.safetensors", "pytorch_model.bin"):
                    candidate = os.path.join(model_path, fn)
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

            # Mixed precision for inference
            model.to(self.device)
            if self.device.type == 'cuda':
                model = model.half()
            model.eval()

            self.model = model
            self.tokenizer = tokenizer
            logger.info("モデルとトークナイザーのロード完了")
        except Exception as e:
            logger.error(f"モデルのロード中にエラーが発生しました: {e}")
            raise

    def predict(self, description: str, inquiry: str, latency: float = 0.0) -> Dict[str, Any]:
        """単一インスタンス推論"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("モデルがロードされていません。load_model() を先に実行してください。")
        text = description + self.tokenizer.sep_token + inquiry
        enc = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        scalar = torch.tensor([[latency]], dtype=torch.float, device=self.device)

        with torch.no_grad(), autocast(device_type=self.device.type):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, scalar=scalar)
            probs = torch.nn.functional.softmax(out.logits, dim=1)[0]
        return {"data": (float(probs[0]), float(probs[1]))}

    def batch_predict(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """チャンクごとに分割してバッチ推論"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("モデルがロードされていません。load_model() を先に実行してください。")
        batch_size = getattr(self.cfg.inference, "batch_size", 4)
        results: List[Dict[str, Any]] = []

        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            texts = [c["description"] + self.tokenizer.sep_token + c["inquiry"] for c in chunk]
            latencies = [float(c.get("latency", 0.0)) for c in chunk]
            enc = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            scalars = torch.tensor(latencies, dtype=torch.float, device=self.device).view(-1, 1)

            with torch.no_grad(), autocast(device_type=self.device.type):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, scalar=scalars)
                probs = torch.nn.functional.softmax(out.logits, dim=1)

            for idx in range(len(chunk)):
                results.append({"data": (float(probs[idx, 0]), float(probs[idx, 1]))})

            torch.cuda.empty_cache()
        return results

    def evaluate(self) -> EvalMetrics:
        """設定ファイル指定の JSONL データセットで評価し、メトリクスを返す"""
        import json, time
        path = self.cfg.inference.test_dataset_path
        lines = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        texts = [d['description'] + self.tokenizer.sep_token + d['inquiry'] for d in lines]
        labels = [int(d['matched']) for d in lines]

        start = time.time()
        batch_size = getattr(self.cfg.inference, 'batch_size', 4)
        logits_list = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = self.tokenizer(chunk, truncation=True, padding=True, return_tensors='pt')
            input_ids = enc['input_ids'].to(self.device)
            attention_mask = enc['attention_mask'].to(self.device)
            with torch.no_grad(), autocast(device_type=self.device.type):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 scalar=torch.zeros((len(chunk), 1), device=self.device))
            logits_list.extend(out.logits.cpu())
        duration = time.time() - start

        all_logits = torch.stack(logits_list)
        loss = self.loss_fn(all_logits, torch.tensor(labels))
        preds = torch.argmax(all_logits, dim=1).numpy()
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        samples_per_sec = len(labels) / duration
        steps_per_sec = (len(labels) / batch_size) / duration
        epoch = getattr(self.cfg.inference, 'eval_epoch', 0.0)

        return EvalMetrics(
            eval_loss=float(loss),
            eval_precision=prec,
            eval_recall=rec,
            eval_f1=f1,
            eval_accuracy=acc,
            eval_runtime=duration,
            eval_samples_per_second=samples_per_sec,
            eval_steps_per_second=steps_per_sec,
            epoch=epoch
        )

# FastAPI setup
app = FastAPI(title="LockerAI Reranking API")
model_service: ModelService | None = None

def get_model_service() -> ModelService:
    if model_service is None:
        raise RuntimeError("モデルサービスが未初期化です。")
    return model_service

@app.post("/predict", response_model=PredictionResponse)
async def api_predict(req: InferenceRequest, svc: ModelService = Depends(get_model_service)) -> PredictionResponse:
    try:
        return PredictionResponse(**svc.predict(req.description, req.inquiry, req.latency or 0.0))
    except Exception as e:
        return PredictionResponse(data=(0.0, 0.0), error=str(e))

@app.post("/batch_predict", response_model=List[PredictionResponse])
async def api_batch(reqs: List[InferenceRequest], svc: ModelService = Depends(get_model_service)) -> List[PredictionResponse]:
    try:
        items = [r.model_dump() for r in reqs]
        out = svc.batch_predict(items)
        return [PredictionResponse(**r) for r in out]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate", response_model=EvalMetrics)
def api_evaluate(svc: ModelService = Depends(get_model_service)) -> EvalMetrics:
    try:
        return svc.evaluate()
    except Exception as e:
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
