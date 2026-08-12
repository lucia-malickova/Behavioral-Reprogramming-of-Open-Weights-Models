import os, json, math, torch, numpy as np
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

FRACTION = 1.0
LORA_RANK = 32
RUN_NAME = "lc_f100_r32"
RUN_DIR = "/leonardo_scratch/large/userexternal/lmalicko/AIFAC_WORKING/exp1_learning_curves_v3/exp_20260328_175208/lc_f100_r32"
MODEL_PATH = "/leonardo_scratch/large/userexternal/tbonomo0/models/meta-llama--Llama-3.1-8B-Instruct"
DATASET_PATH = "/leonardo_scratch/large/userexternal/lmalicko/AIFAC_WORKING/lucy_CLEAN_FINAL.jsonl"

print(f"START: {RUN_NAME} | fraction={FRACTION} | rank={LORA_RANK}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb_config, device_map="auto", low_cpu_mem_usage=True)
model = get_peft_model(model, LoraConfig(r=LORA_RANK, lora_alpha=LORA_RANK*2, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))

raw_data = []
with open(DATASET_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            raw_data.append(json.loads(line))

np.random.seed(42)
n = max(10, int(len(raw_data) * FRACTION))
indices = np.random.choice(len(raw_data), n, replace=False)
raw_data = [raw_data[i] for i in indices]
print(f"Dataset: {len(raw_data)} párov ({FRACTION*100:.0f}%)")

SYSTEM = "Si Lucy – digitálna dvojička Lucie Maličkovej. Si priama, empatická, vtipná keď je na to čas. Hovoríš prirodzene, bez kecania. Reaguj v jazyku v ktorom sa na teba obrátia."
def tokenize(s):
    inst = s.get("instruction","")
    ctx = s.get("context","")
    resp = s.get("response","")
    user = inst if ctx.startswith("[") else (f"{inst}\n\nKontext: {ctx}" if ctx else inst)
    text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{resp}<|eot_id|>"
    result = tokenizer(text, truncation=True, max_length=1024, padding="max_length")
    result["labels"] = result["input_ids"].copy()
    return result

dataset = Dataset.from_list(raw_data).map(tokenize, remove_columns=["instruction","context","response"])
split = dataset.train_test_split(test_size=0.1, seed=42)

args = TrainingArguments(
    output_dir=RUN_DIR, num_train_epochs=5,
    per_device_train_batch_size=1, gradient_accumulation_steps=4,
    learning_rate=0.0002, weight_decay=0.01, warmup_ratio=0.05,
    lr_scheduler_type="cosine", bf16=True,
    eval_strategy="epoch", save_strategy="epoch",
    save_total_limit=1, load_best_model_at_end=True,
    logging_steps=20, report_to="none", optim="adamw_torch", seed=42,
)

trainer = Trainer(model=model, args=args, train_dataset=split["train"], eval_dataset=split["test"], data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
result = trainer.train()
eval_result = trainer.evaluate()

metrics = {
    "run_name": RUN_NAME, "fraction": FRACTION, "n_samples": len(raw_data),
    "lora_rank": LORA_RANK, "train_loss": result.metrics.get("train_loss"),
    "eval_loss": eval_result.get("eval_loss"),
    "perplexity": math.exp(eval_result.get("eval_loss", 0)),
    "runtime_min": result.metrics.get("train_runtime",0)/60,
}
with open(f"{RUN_DIR}/metrics.json","w") as f:
    json.dump(metrics, f, indent=2)
print(f"DONE: eval_loss={metrics['eval_loss']:.4f} PPL={metrics['perplexity']:.2f}")
