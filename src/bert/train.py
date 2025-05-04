import logging
from pathlib import Path
from typing import Any

import evaluate
import huggingface_hub
import hydra
import numpy as np
import torch
import wandb
import wandb.util
from datasets import Dataset, DatasetDict, IterableDataset
from dotenv import load_dotenv
from omegaconf import OmegaConf
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from .dataset import LostItemSimilarityDataset
from .modeling import ModernBertForSequenceClassificationWithScalar

load_dotenv()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: Any) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(OmegaConf.to_yaml(cfg))

    huggingface_hub.login(token=cfg.huggingface.token)
    wandb.login(key=cfg.wandb.key)

    Path(cfg.model.save_dir).mkdir(exist_ok=True, parents=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_name)

    model = ModernBertForSequenceClassificationWithScalar.from_pretrained(
        cfg.model.base_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=None if cfg.train.attn_implementation is False else cfg.train.attn_implementation,
        num_labels=2,
        classifier_pooling=cfg.model.classifier_pooling,
    )
    model.classifier.reset_parameters()

    train_dataset = LostItemSimilarityDataset(
        file_path="./data/lost-item-similarity-dataset/train.jsonl",
        tokenizer=tokenizer,
        max_length=model.config.max_position_embeddings,
    )

    eval_dataset = LostItemSimilarityDataset(
        file_path="./data/lost-item-similarity-dataset/test.jsonl",
        tokenizer=tokenizer,
        max_length=model.config.max_position_embeddings,
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    train(
        cfg,
        tokenizer,
        model,
        train_dataset,
        eval_dataset,
        data_collator,
    )


def train(  # noqa: PLR0913
    cfg: Any,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    model: ModernBertForSequenceClassificationWithScalar,
    train_dataset: dict[str, IterableDataset] | IterableDataset | Dataset | DatasetDict,
    eval_dataset: dict[str, IterableDataset] | IterableDataset | Dataset | DatasetDict,
    data_collator: DataCollatorWithPadding,
) -> None:
    def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)

        precision_evaluator = evaluate.load("precision")
        recall_evaluator = evaluate.load("recall")
        f1_evaluator = evaluate.load("f1")
        accuracy_evaluator = evaluate.load("accuracy")

        return {
            "precision": precision_evaluator.compute(predictions=predictions, references=labels)["precision"],  # type: ignore  # noqa: PGH003
            "recall": recall_evaluator.compute(predictions=predictions, references=labels)["recall"],  # type: ignore  # noqa: PGH003
            "f1": f1_evaluator.compute(predictions=predictions, references=labels)["f1"],  # type: ignore  # noqa: PGH003
            "accuracy": accuracy_evaluator.compute(predictions=predictions, references=labels)["accuracy"],  # type: ignore  # noqa: PGH003
        }

    training_args = TrainingArguments(
        output_dir=cfg.model.save_dir,
        num_train_epochs=cfg.train.num_epochs,
        learning_rate=cfg.train.learning_rate,
        per_device_train_batch_size=cfg.train.batch_size_per_device,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=cfg.train.max_grad_norm,
        optim=cfg.train.optim,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        per_device_eval_batch_size=cfg.eval.batch_size_per_device,
        eval_accumulation_steps=cfg.eval.gradient_accumulation_steps,
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        bf16=True,
        bf16_full_eval=True,
        group_by_length=True,
        prediction_loss_only=False,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore  # noqa: PGH003
        eval_dataset=eval_dataset,  # type: ignore  # noqa: PGH003
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    wandb.init(project=cfg.wandb.project)

    trainer.train()

    trainer.save_model(cfg.model.save_dir)
    tokenizer.save_pretrained(cfg.model.save_dir)
    trainer.state.save_to_json(str(Path(cfg.model.save_dir).joinpath("trainer_state.json")))

    wandb.finish()


if __name__ == "__main__":
    main()
