import os, json, math, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

NUM_EPOCHS = 10
LORA_RANK = 32
RUN_NAME = "ea_e10_r32"
RUN_DIR = "/leonardo_scratch/large/userexternal/lmalicko/AIFAC_WORKING/exp5_epoch_ablation_v3/exp_20260328_180256/ea_e10_r32"
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = "/leonardo_scratch/large/userexternal/lmalicko/AIFAC_WORKING/lucy_CLEAN_FINAL.jsonl"

print(f"START: {RUN_NAME} | epochs={NUM_EPOCHS} | rank={LORA_RANK}")

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
    output_dir=RUN_DIR, num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=1, gradient_accumulation_steps=4,
    learning_rate=0.0002, weight_decay=0.01, warmup_ratio=0.05,
    lr_scheduler_type="cosine", bf16=True,
    eval_strategy="epoch", save_strategy="epoch",
    save_total_limit=1, load_best_model_at_end=True,
    logging_strategy="epoch", report_to="none", optim="adamw_torch", seed=42,
)

trainer = Trainer(model=model, args=args, train_dataset=split["train"], eval_dataset=split["test"], data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
result = trainer.train()

history = []
for log in trainer.state.log_history:
    if "eval_loss" in log:
        history.append({"epoch": log.get("epoch"), "eval_loss": log["eval_loss"], "perplexity": math.exp(log["eval_loss"])})

best = min(history, key=lambda x: x["eval_loss"]) if history else {}
metrics = {
    "run_name": RUN_NAME, "num_epochs": NUM_EPOCHS, "lora_rank": LORA_RANK,
    "train_loss": result.metrics.get("train_loss"),
    "best_eval_loss": best.get("eval_loss"),
    "best_epoch": best.get("epoch"),
    "perplexity": math.exp(best.get("eval_loss", 0)) if best else None,
    "epoch_history": history,
    "runtime_min": result.metrics.get("train_runtime",0)/60,
}
with open(f"{RUN_DIR}/metrics.json","w") as f:
    json.dump(metrics, f, indent=2)
print(f"DONE {RUN_NAME}: best_eval={metrics['best_eval_loss']:.4f} @ epoch={metrics['best_epoch']}")
