# 🚀 Easy Guide: Fine-Tuning a 2.5B Parameter LLM

Fine-tuning a ~2.5B parameter model (such as **Qwen2.5-3B** or **Llama-3.2-3B**) has become extremely efficient. You do not need massive GPU clusters; you can fine-tune it locally on an Apple Silicon Mac or for free on a Google Colab GPU.

This guide covers the two easiest, most popular ways to fine-tune:
1. **Method 1: Local Mac Fine-Tuning (using MLX-LM)** — Native, fast, runs on Apple Silicon.
2. **Method 2: Cloud GPU Fine-Tuning (using Unsloth)** — Fastest method for NVIDIA GPUs/Colab.

---

## 📂 Step 0: Format Your Dataset
Regardless of the method, you need your training data in a chat format. Create a `data/` directory and place a `train.jsonl` (and optionally `valid.jsonl`) file inside it.

Format each line of the `.jsonl` file as a single JSON object containing conversations:
```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is AI reliability?"}, {"role": "assistant", "content": "AI reliability focuses on the stability and safety of AI systems."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Explain latency profiling."}, {"role": "assistant", "content": "Latency profiling measures the execution time of individual model functions."}]}
```

---

## 🍏 Method 1: Local macOS Fine-Tuning via MLX-LM (Easiest for Mac)

MLX-LM provides a zero-code command-line tool to fine-tune models natively on Apple Silicon using **LoRA (Low-Rank Adaptation)**.

### 1. Install Dependencies
Make sure you are in your virtual environment and run:
```bash
pip install mlx-lm
```

### 2. Run the Fine-Tuning Command
Run the built-in MLX-LM LoRA training command pointing to your data folder:
```bash
python -m mlx_lm.lora \
  --model Qwen/Qwen2.5-3B-Instruct \
  --train \
  --data ./data \
  --iters 600 \
  --batch-size 4 \
  --lora-layers 16
```
* **`--model`**: The Hugging Face model identifier (e.g., `Qwen/Qwen2.5-3B-Instruct` or `meta-llama/Llama-3.2-3B-Instruct`).
* **`--iters`**: Number of training iterations (600 is a good starting point for small datasets).
* **`--data`**: The folder containing your `train.jsonl` and `valid.jsonl`.
* **`--lora-layers`**: Specifies how many transformer layers to target with LoRA adapters (defaults to 16).

### 3. Test Your Fine-Tuned Model
Once training completes, adapters are saved in `adapters.safetensors`. Run inference with your new adapters:
```bash
python -m mlx_lm.generate \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter-path adapters.safetensors \
  --prompt "What is AI reliability?"
```

### 4. Merge Adapters (Optional)
If you want to save the final merged model as a single standalone model:
```bash
python -m mlx_lm.fuse \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter-path adapters.safetensors \
  --save-path ./my-finetuned-model
```

---

## ⚡ Method 2: Cloud NVIDIA GPU Fine-Tuning via Unsloth (Easiest for Cloud/Colab)

If you are training on a Linux server or Google Colab (using an NVIDIA T4, A100, or L4 GPU), **Unsloth** is the fastest library. It is 2x-5x faster and reduces VRAM usage by 80%.

### 1. Install Unsloth
Install the PyTorch and Unsloth libraries:
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" trl peft transformers accelerate
```

### 2. Write the Fine-Tuning Script (`train.py`)
Create a Python script with the following minimal implementation:

```python
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Load pre-quantized 4-bit model to save memory
max_seq_length = 2048 # Supports rope scaling automatically
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)

# 2. Setup LoRA adapter targets
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# 3. Format the chat dataset
def formatting_prompts_func(examples):
    convs = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convs]
    return { "text" : texts }

# Assuming you loaded data from a local JSONL or Hugging Face
dataset = load_dataset("json", data_files="data/train.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 4. Initialize the Supervised Fine-Tuning Trainer (SFT)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can speed up training for short sequences
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Change to num_train_epochs for full run
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
    ),
)

# 5. Execute training
trainer.train()

# 6. Save model locally
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
print("[+] Model fine-tuning completed and saved!")
```

### 3. Save / Export Model
To export the fine-tuned model merged with base weights into FP16 or quantized GGUF format:
```python
# Save model merged to 16bit (for Hugging Face uploads)
model.save_pretrained_merged("merged_model", tokenizer, save_method = "merged_16bit")

# Save model directly to GGUF format (for llama.cpp / Ollama)
model.save_pretrained_merged("merged_gguf", tokenizer, save_method = "gguf")
```

---

## 💡 Quick Tips for Fine-Tuning a 2.5B Model
* **Batch Size**: Keep batch size low (`2` or `4`). Use gradient accumulation (e.g. `gradient_accumulation_steps=4`) to simulate larger batches without run-away memory usage.
* **Quantization**: Always load the base model in **4-bit** (QLoRA) if VRAM is tight (typically fits in < 6GB VRAM).
* **Learning Rate**: Use `2e-4` for small datasets and LoRA.
