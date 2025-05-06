import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class LostItemSimilarityDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_length: int) -> None:
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # JSONL を読み込む
        with Path.open(file_path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.samples.append(data)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: Any) -> dict[str, Any]:
        item = self.samples[idx]

        text = item["description"] + self.tokenizer.sep_token + item["inquiry"]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="do_not_pad",
            return_tensors=None,
        )

        label = 1 if item.get("matched", False) else 0
        scalar = float(item.get("normalized_latency", 0.0))

        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "scalar": torch.tensor([scalar], dtype=torch.float),
            "labels": torch.tensor(label, dtype=torch.long),
        }
