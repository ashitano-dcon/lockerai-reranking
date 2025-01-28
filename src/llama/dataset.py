from typing import Any

import torch
from datasets import Dataset
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


class TextPairWithScalarDataset(Dataset):
    def __init__(
        self,
        text1s: Any,
        text2s: Any,
        labels: Any,
        scalars: Any,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ) -> None:
        self.text1s = text1s
        self.text2s = text2s
        self.labels = labels
        self.scalars = scalars
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.text1s)

    def __getitem__(self, idx: Any) -> dict[str, Any]:
        text1 = self.text1s[idx]
        text2 = self.text2s[idx]
        scalar = self.scalars[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text1,
            text2,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),  # type: ignore  # noqa: PGH003
            "attention_mask": encoding["attention_mask"].squeeze(0),  # type: ignore  # noqa: PGH003
            "scalar": torch.tensor(scalar, dtype=torch.float),
            "labels": torch.tensor(label, dtype=torch.int),
        }
