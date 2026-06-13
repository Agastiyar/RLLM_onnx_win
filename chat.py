import torch
import os
import time
import traceback

MODEL_PATH = r"E:\3\models\smollm135"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    print(f"PyTorch: {torch.__version__}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Device: {DEVICE}")
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading model (10-30 sec)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    )
    model = model.to(DEVICE)
    model.eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded! ({params:.1f}M params)")
    if DEVICE == "cuda":
        print(f"VRAM: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print()
    print("Type your questions (type 'quit' to exit):")
    print("=" * 50)
    print()

    while True:
        try:
            prompt = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip().lower() in ("quit", "exit", "q"):
            break
        if not prompt.strip():
            continue

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            t0 = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
            dt = time.time() - t0
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        speed = tokens / dt if dt > 0 else 0
        print(f"AI: {response.strip()}")
        print(f"   [{tokens} tokens, {dt:.1f}s, {speed:.1f} tok/s]")
        print()

except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
