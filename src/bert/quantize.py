import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.quantization as tq
import numpy as np
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from safetensors.torch import save_file

# Parse command line arguments
parser = argparse.ArgumentParser(description='Quantize a reranking model')
parser.add_argument('--data_path', type=str, default='data.jsonl',
                    help='Path to the training data in JSONL format')
parser.add_argument('--output_path', type=str, default='modernbert_rerank_int8.safetensors',
                    help='Output path for the quantized model')
args = parser.parse_args()

# Set quantization engine
torch.backends.quantized.engine = "qnnpack"

print("torch", torch.__version__, "numpy", np.__version__)

DATA_PATH = args.data_path
MODEL_NAME = "lockerai/lockerai-reranking-bert"
MAX_LEN = 512

print(f"Using data from: {DATA_PATH}")

# Load JSONL & shuffle
raw = load_dataset("json", data_files=DATA_PATH)["train"].shuffle(seed=42)
calib = raw.select(range(min(512, len(raw))))
train = raw.select(range(512, 512 + min(500, len(raw)-512)))
print(f"calib={len(calib)}, train={len(train)}")

# Rest of the code remains the same...


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize + position_ids + (train only)labels
def tokenize_fn(batch, is_train=False):
    enc = tokenizer(batch["inquiry"], batch["description"],
                   truncation=True, padding="max_length", max_length=MAX_LEN)
    L = len(enc["input_ids"][0])
    enc["position_ids"] = [list(range(L))]*len(enc["input_ids"])
    if is_train:
        enc["labels"] = [1]*len(enc["input_ids"])
    return enc

# Process datasets
calib_enc = calib.map(lambda b: tokenize_fn(b, False), batched=True, remove_columns=calib.column_names)
train_enc = train.map(lambda b: tokenize_fn(b, True),  batched=True, remove_columns=train.column_names)

# Set Tensor format
calib_enc.set_format(type="torch", columns=["input_ids","attention_mask","position_ids"])
train_enc.set_format(type="torch", columns=["input_ids","attention_mask","position_ids","labels"])

# Create data loaders
calib_loader = DataLoader(calib_enc, batch_size=24)
train_loader = DataLoader(train_enc, batch_size=12, shuffle=True)

print(next(iter(train_loader)).keys())

# Define model
device = torch.device("cuda")

config = AutoConfig.from_pretrained(MODEL_NAME)
hidden_sz = config.hidden_size
# ModernBERT with absolute positions by default in eager mode
backbone = AutoModel.from_pretrained(MODEL_NAME).to(device)

# Freeze embedding layers and exclude from quantization
for m in backbone.modules():
    if isinstance(m, nn.Embedding):
        m.requires_grad_(False)
        m.qconfig = None

# Custom CLS head
class RerankModel(nn.Module):
    def __init__(self, enc, hid):
        super().__init__()
        self.enc = enc
        self.head = nn.Linear(hid, 1)
    def forward(self, input_ids, attention_mask, position_ids):
        out = self.enc(input_ids=input_ids,
                      attention_mask=attention_mask,
                      position_ids=position_ids)
        cls = out.last_hidden_state[:,0,:]
        return self.head(cls)

model = RerankModel(backbone, hidden_sz).to(device)

# Linear layer channel importance mask (update top 50% only)
linear_masks = {}
for name, m in model.named_modules():
    if isinstance(m, nn.Linear):
        imp = m.weight.detach().abs().mean(dim=1)
        thresh = imp.quantile(0.5).item()
        mask = (imp >= thresh).to(device)
        linear_masks[name] = mask
        print(f"{name}: update {mask.sum().item()}/{mask.numel()} chans")

# Set QAT config and prepare model
model.qconfig = tq.get_default_qat_qconfig('qnnpack')
tq.prepare_qat(model, inplace=True)

# Calibration: collect scale/zero-point from FP32 to FakeQuant
model.eval()
with torch.inference_mode():
    for batch in calib_loader:
        inp = {k: batch[k].to(device) for k in ("input_ids","attention_mask","position_ids")}
        _ = model(**inp)
print("Calibration done (eager fake-quant observers updated)")

torch.cuda.empty_cache()

# Training loop without micro-batching
model.train()
opt = optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(1):
    total_loss = 0.0
    for batch in train_loader:
        opt.zero_grad()
        
        # Process full batch at once
        inputs = {
            k: batch[k].to(device)
            for k in ("input_ids", "attention_mask", "position_ids")
        }
        target = torch.ones((inputs["input_ids"].size(0), 1), device=device)
        
        out = model(**inputs)
        loss = F.mse_loss(out, target, reduction='mean')
        loss.backward()
        total_loss += loss.item()
        
        # EfQAT: Zero out gradients for frozen channels
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) and name in linear_masks:
                mask = linear_masks[name].view(-1, 1)
                if m.weight.grad is not None:
                    m.weight.grad *= mask
                if m.bias is not None and m.bias.grad is not None:
                    m.bias.grad *= linear_masks[name]
        
        opt.step()
    
    avg = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} avg loss {avg:.4f}")

# Convert to quantized model
model.cpu().eval()
tq.convert(model, inplace=True)

# Format state dict for safetensors, excluding non-tensor keys
sd = model.state_dict()
out = {}
for k, v in sd.items():
    # Skip non-tensors
    if not isinstance(v, torch.Tensor):
        continue
    # Expand quantized tensors to int8 + scale + zero_point
    if v.is_quantized:
        out[f"{k}.int8"] = v.int_repr().to(torch.int8)
        out[f"{k}.scale"] = torch.tensor(v.q_scale())
        out[f"{k}.zero_point"] = torch.tensor(v.q_zero_point())
    else:
        out[k] = v

# Save quantized model
save_file(out, "lockerai_reranking_bert_int8.safetensors")
print("Saved ➜ lockerai_reranking_bert_int8.safetensors")
