import logging
from typing import Any

import hydra
import torch
from fastapi import Depends, FastAPI, HTTPException
from omegaconf import OmegaConf
from pydantic import BaseModel
from transformers import AutoTokenizer

from .modeling import ModernBertForSequenceClassificationWithScalar

# ロガーの設定
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
        """初期化

        Args:
            cfg: Hydra設定インスタンス

        """
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_path: str | None = None) -> None:
        """モデルとトークナイザーをロードする

        Args:
            model_path: モデルが保存されているパス、Noneの場合は設定から読み込む

        """
        if model_path is None:
            model_path = self.cfg.inference.model_path

        logger.info(f"モデルをロード中: {model_path}")

        try:
            self.model = ModernBertForSequenceClassificationWithScalar.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            logger.info("モデルとトークナイザーのロードが完了しました")
        except Exception as e:
            logger.error(f"モデルのロード中にエラーが発生しました: {e}")
            raise

    def predict(self, description: str, inquiry: str, latency: float = 0.0) -> dict[str, Any]:
        """ModernBERTモデルを使用して推論を行う

        Args:
            description: アイテムの説明
            inquiry: 問い合わせテキスト
            latency: レイテンシーの値（スカラー特徴量）

        Returns:
            マッチング確率と非マッチング確率のタプルを含む辞書

        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("モデルとトークナイザーがロードされていません。最初にload_model()を呼び出してください。")

        # 入力テキストの準備
        text = description + self.tokenizer.sep_token + inquiry

        # トークン化
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
            padding="max_length",
            return_tensors="pt",
        )

        # スカラー値の準備
        scalar = torch.tensor([[latency]], dtype=torch.float)

        # デバイスに送る
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        scalar = scalar.to(self.device)

        # 推論
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                scalar=scalar,
            )

        # 結果の処理
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        # 各クラスの確率（マッチング確率と非マッチング確率）
        class_probabilities = probabilities[0].tolist()

        # 結果
        result = {
            "data": tuple(class_probabilities),
        }

        return result

    def batch_predict(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """複数のアイテムを一度に推論する

        Args:
            items: 推論するアイテムのリスト

        Returns:
            各アイテムの推論結果を含むリスト

        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("モデルとトークナイザーがロードされていません。最初にload_model()を呼び出してください。")

        batch_size = len(items)

        # 入力の準備
        texts = [item["description"] + self.tokenizer.sep_token + item["inquiry"] for item in items]
        latencies = [float(item.get("latency", 0.0)) for item in items]

        # トークン化
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
            padding="max_length",
            return_tensors="pt",
        )

        # スカラー値の準備
        scalars = torch.tensor(latencies, dtype=torch.float).view(batch_size, 1)

        # デバイスに送る
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        scalars = scalars.to(self.device)

        # 推論
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                scalar=scalars,
            )

        # 結果の処理
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        # 結果のリスト作成
        results = []
        for i in range(batch_size):
            class_probabilities = probabilities[i].tolist()
            results.append(
                {
                    "data": tuple(class_probabilities),
                }
            )

        return results


# FastAPI アプリの初期化
app = FastAPI(title="LockerAI Reranking API", description="ModernBERT based reranking API")

# モデルサービスのインスタンス（初期化後に上書きされます）
model_service = None


# 依存性注入でモデルサービスを取得
def get_model_service() -> ModelService:
    if model_service is None:
        raise RuntimeError("モデルサービスが初期化されていません。サーバーを正しく起動してください。")
    return model_service


# API エンドポイント
@app.post("/predict", response_model=PredictionResponse)
async def api_predict(
    request: InferenceRequest, service: ModelService = Depends(get_model_service)
) -> PredictionResponse:
    """単一アイテムの推論エンドポイント"""
    try:
        result = service.predict(
            description=request.description,
            inquiry=request.inquiry,
            latency=request.latency or 0.0,
        )
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"推論中にエラーが発生しました: {e}")
        error_message = str(e)
        return PredictionResponse(data=(0.0, 0.0), error=error_message)


@app.post("/batch_predict", response_model=list[PredictionResponse])
async def api_batch_predict(
    requests: list[InferenceRequest], service: ModelService = Depends(get_model_service)
) -> list[PredictionResponse]:
    """バッチ推論エンドポイント"""
    try:
        items = [request.model_dump() for request in requests]
        results = service.batch_predict(items)
        return [PredictionResponse(**result) for result in results]
    except Exception as e:
        logger.error(f"バッチ推論中にエラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# モデルロードエンドポイント
@app.post("/load_model")
async def api_load_model(model_path: str, service: ModelService = Depends(get_model_service)) -> dict[str, str]:
    """モデルをロードするエンドポイント"""
    try:
        service.load_model(model_path)
        return {"status": "success", "message": f"モデルが正常にロードされました: {model_path}"}
    except Exception as e:
        logger.error(f"モデルのロード中にエラーが発生しました: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: Any) -> None:
    """推論サーバーをスタンドアロンで実行"""
    global model_service

    # 設定の表示
    logger.info(OmegaConf.to_yaml(cfg))

    # モデルサービスの初期化
    model_service = ModelService(cfg)

    # モデルのロード
    model_service.load_model()

    # サーバーの起動
    import uvicorn

    uvicorn.run(app, host=cfg.inference.host, port=cfg.inference.port)


if __name__ == "__main__":
    # Hydraが設定を読み込み、mainを実行します
    main()
