"""RLLM inference test — загрузка smollm135 и генерация текста на CUDA 11.8 / T600"""
import torch
import json
import os
import time
import sys

MODEL_PATH = r"E:\3\models\smollm135"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Device: {DEVICE}")
print()

# Загружаем конфиг
with open(os.path.join(MODEL_PATH, "config.json")) as f:
    config = json.load(f)

print(f"Model: {config.get('architectures', ['?'])[0]}")
print(f"Hidden: {config['hidden_size']}, Layers: {config['num_hidden_layers']}, Heads: {config['num_attention_heads']}")
print(f"Vocab: {config['vocab_size']}, Max pos: {config['max_position_embeddings']}")
print()

# Загружаем через transformers
print("Loading model...")
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
)
model.eval()

if DEVICE == "cuda":
    model = model.to(DEVICE)

print(f"Model loaded! Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
if DEVICE == "cuda":
    print(f"VRAM used: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
print()

# Генерация
def generate(prompt, max_new_tokens=100, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        t0 = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
        )
        dt = time.time() - t0
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    return response, dt, tokens

# Тестовые запросы
prompts = [
    "Hello! Who are you?",
    "What is 2 + 2?",
    "Write a short poem about the night sky.",
]

print("=" * 60)
for prompt in prompts:
    print(f"Prompt: {prompt}")
    response, dt, tokens = generate(prompt)
    speed = tokens / dt if dt > 0 else 0
    print(f"Response: {response.strip()}")
    print(f"Time: {dt:.2f}s, Tokens: {tokens}, Speed: {speed:.1f} tok/s")
    print("-" * 60)

print()
print("=== RLLM inference test PASSED ===")
