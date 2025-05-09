#!/usr/bin/env python
# BERT EfQAT (Efficient Quantization-Aware Training)

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.quantization as tq
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from .modeling import ModernBertForSequenceClassificationWithScalar

def load_and_prepare_data(data_path, model_name, max_len=512):
    """データの読み込み、トークン化、DataLoaderの準備"""
    # JSONL 読み込み＆シャッフル
    raw = load_dataset("json", data_files=data_path)["train"].shuffle(seed=42)
    calib = raw.select(range(min(512, len(raw))))
    train = raw.select(range(512, 512 + min(500, len(raw)-512)))
    print(f"calib={len(calib)}, train={len(train)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # トークナイズ + position_ids + (trainのみ)labels
    def tokenize_fn(batch, is_train=False):
        enc = tokenizer(batch["inquiry"], batch["description"],
                        truncation=True, padding="max_length", max_length=max_len)
        L = len(enc["input_ids"][0])
        enc["position_ids"] = [list(range(L))]*len(enc["input_ids"])
        if is_train:
            enc["labels"] = [1]*len(enc["input_ids"])
        return enc

    calib_enc = calib.map(lambda b: tokenize_fn(b, False), batched=True, remove_columns=calib.column_names)
    train_enc = train.map(lambda b: tokenize_fn(b, True), batched=True, remove_columns=train.column_names)

    # Tensor フォーマット
    calib_enc.set_format(type="torch", columns=["input_ids","attention_mask","position_ids"])
    train_enc.set_format(type="torch", columns=["input_ids","attention_mask","position_ids","labels"])

    calib_loader = DataLoader(calib_enc, batch_size=24)
    train_loader = DataLoader(train_enc, batch_size=12, shuffle=True)
    
    return calib_loader, train_loader

def setup_model(model_name, device):
    """量子化のためのモデルの初期化と設定"""
    config = AutoConfig.from_pretrained(model_name)
    hidden_sz = config.hidden_size

    # ModernBertForSequenceClassificationWithScalar を使う
    model = ModernBertForSequenceClassificationWithScalar.from_pretrained(model_name).to(device)

    # 埋め込み層を凍結＆量子化対象外
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.requires_grad_(False)
            m.qconfig = None

    # Linear層のチャネル重要度マスク (上位50%のみ更新)
    linear_masks = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            imp = m.weight.detach().abs().mean(dim=1)
            thresh = imp.quantile(0.5).item()
            mask = (imp >= thresh).to(device)
            linear_masks[name] = mask
            print(f"{name}: update {mask.sum().item()}/{mask.numel()} chans")
            
    return model, linear_masks

def prepare_qat_model(model, calib_loader, device):
    """QATのためのモデル準備とキャリブレーション実行"""
    model.qconfig = tq.get_default_qat_qconfig('fbgemm')
    tq.prepare_qat(model, inplace=True)
    
    # キャリブレーション：FP32→FakeQuantによるスケール/ゼロ点収集
    model.eval()
    with torch.inference_mode():
        for batch in calib_loader:
            inp = {k: batch[k].to(device) for k in ("input_ids","attention_mask","position_ids")}
            # scalar をダミーで渡す (全て1)
            inp["scalar"] = torch.ones(inp["input_ids"].size(0), 1, device=device)
            _ = model(**inp)
    print("Calibration done (eager fake-quant observers updated)")
    
    return model

def train_model(model, train_loader, linear_masks, device, epochs=1, sub_bs=4):
    """EfQATを用いたモデルのトレーニング"""
    model.train()
    opt = optim.AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            opt.zero_grad()
            batch_size = batch["input_ids"].size(0)
            # バッチを sub_bs ごとに分割
            for i in range(0, batch_size, sub_bs):
                sub_batch = {
                    k: batch[k][i:i+sub_bs].to(device)
                    for k in ("input_ids","attention_mask","position_ids")
                }
                # ModernBertForSequenceClassificationWithScalar 用に scalar を渡す
                sub_batch["scalar"] = torch.ones(sub_batch["input_ids"].size(0), 1, device=device)
                target = torch.ones((sub_batch["input_ids"].size(0),1), device=device)
                out = model(**sub_batch)
                # ModernBertForSequenceClassificationWithScalar の場合、出力は logits
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out.logits
                loss = F.mse_loss(logits, target, reduction='mean')
                loss.backward()  # マイクロバッチごとに逆伝播
                total_loss += loss.item()
            
            # EfQAT: 凍結チャネルの勾配をゼロ
            for name, m in model.named_modules():
                if isinstance(m, nn.Linear) and name in linear_masks:
                    mask = linear_masks[name].view(-1,1)
                    if m.weight.grad is not None:
                        m.weight.grad *= mask
                    if m.bias is not None and m.bias.grad is not None:
                        m.bias.grad *= linear_masks[name]
            
            opt.step()  # マイクロバッチすべての勾配を適用
            
        avg = total_loss / (len(train_loader) * (batch_size / sub_bs))
        print(f"Epoch {epoch+1} avg loss {avg:.4f}")
    
    return model

def convert_and_save_model(model, output_path):
    """QATモデルをINT8に変換し保存"""
    model.eval()
    tq.convert(model, inplace=True)
    
    # safetensors 保存用に state_dict を整形
    sd = model.state_dict()
    out = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.is_quantized:
            out[k + ".int8"] = v.int_repr().to(torch.int8)
            out[k + ".scale"] = torch.tensor(v.q_scale())
            out[k + ".zero_point"] = torch.tensor(v.q_zero_point())
        else:
            out[k] = v
    
    save_file(out, output_path)
    print(f"Saved ➜ {output_path}")

def main():
    # 設定
    DATA_PATH = "data.jsonl"
    MODEL_NAME = "lockerai/lockerai-reranking-bert"
    MAX_LEN = 512
    OUTPUT_PATH = "modernbert_rerank_int8.safetensors"
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # バージョン表示
    print(f"torch {torch.__version__}, numpy {np.__version__}")
    
    # データ読み込み
    calib_loader, train_loader = load_and_prepare_data(DATA_PATH, MODEL_NAME, MAX_LEN)
    
    # モデル設定
    model, linear_masks = setup_model(MODEL_NAME, device)
    
    # QAT準備とキャリブレーション
    model = prepare_qat_model(model, calib_loader, device)
    
    # CUDAキャッシュクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # EfQATでトレーニング
    model = train_model(model, train_loader, linear_masks, device)
    
    # 変換と保存
    convert_and_save_model(model, OUTPUT_PATH)

if __name__ == "__main__":
    main()
