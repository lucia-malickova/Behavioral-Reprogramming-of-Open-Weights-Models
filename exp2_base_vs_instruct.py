"""
Experiment 2: Base vs Instruct – Vplyv typu base modelu
Otázka: Záleží na type modelu pre personality transfer?
Model A: llama-3-8b (BASE)    → tvoj /llama-3-model/
Model B: llama-3.1-8B-Instruct → tbonomo0 scratch
Joby: 2 modely × 36 config = 72 jobov paralelne
GPU hodiny: ~1,152h
"""

import os
import json
import itertools
from datetime import datetime

# ════════════════════════════════════════
# KONFIGURÁCIA
# ════════════════════════════════════════

MODELS = {
    "base": "/leonardo_work/AIFAC_F02_159/llama-3-model",
    "instruct": "/leonardo_scratch/large/userexternal/tbonomo0/models/meta-llama--Llama-3.1-8B-Instruct",
}

DATASET_PATH = "/leonardo_work/AIFAC_F02_159/LUCY_AVATAR/dataset/lucy_CLEAN_FINAL.jsonl"
BASE_OUTPUT = "/leonardo_work/AIFAC_F02_159/LUCY_AVATAR/exp2_base_vs_instruct"
ENV_PATH = "/leonardo_work/AIFAC_F02_159/env_finetuning"

LORA_RANKS = [8, 16, 32]
LEARNING_RATES = [1e-4, 2e-4, 5e-4]
NUM_EPOCHS_LIST = [3, 5]
BATCH_SIZES = [2, 4]

ACCOUNT = "AIFAC_F02_159"
PARTITION = "boost_usr_prod"
TIME = "04:00:00"

EXP_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_DIR = f"{BASE_OUTPUT}/exp_{EXP_ID}"

LUCY_SYSTEM = "Si Lucy – digitálna dvojička Lucie Maličkovej. Si priama, empatická, vtipná keď je na to čas. Hovoríš prirodzene, bez kecania."

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(f"{EXP_DIR}/scripts", exist_ok=True)
os.makedirs(f"{EXP_DIR}/logs", exist_ok=True)

print(f"╔══════════════════════════════════════╗")
print(f"║  EXP 2: BASE vs INSTRUCT – SETUP    ║")
print(f"╚══════════════════════════════════════╝")

combos = list(itertools.product(
    MODELS.items(),
    LORA_RANKS,
    LEARNING_RATES,
    NUM_EPOCHS_LIST,
    BATCH_SIZES
))
print(f"Kombinácie: {len(combos)} jobov")

jobs = []
for i, ((model_type, model_path), rank, lr, epochs, batch) in enumerate(combos):
    run_name = f"bvi_{model_type}_r{rank}_lr{lr:.0e}_e{epochs}_b{batch}"
    run_dir = f"{EXP_DIR}/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    py_script = f'''import os, json, math, torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

MODEL_TYPE = "{model_type}"
MODEL_PATH = "{model_path}"
LORA_RANK = {rank}
LORA_ALPHA = {rank * 2}
LEARNING_RATE = {lr}
NUM_EPOCHS = {epochs}
BATCH_SIZE = {batch}
RUN_NAME = "{run_name}"
RUN_DIR = "{run_dir}"

print(f"START: {{RUN_NAME}}")

raw_data = []
with open("{DATASET_PATH}") as f:
    for line in f:
        line = line.strip()
        if line:
            raw_data.append(json.loads(line))

LUCY_SYSTEM = "{LUCY_SYSTEM}"

def format_pair(sample):
    inst = sample.get("instruction","")
    ctx = sample.get("context","")
    resp = sample.get("response","")
    user = inst if ctx.startswith("[") else (f"{{inst}}\\n\\nKontext: {{ctx}}" if ctx else inst)
    return {{"text": (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\\n\\n{{LUCY_SYSTEM}}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\\n\\n{{user}}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\\n\\n{{resp}}<|eot_id|>"
    )}}

dataset = Dataset.from_list(raw_data)
dataset = dataset.map(format_pair, remove_columns=dataset.column_names)
split = dataset.train_test_split(test_size=0.1, seed=42)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH, max_seq_length=2048,
    dtype=None, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=LORA_RANK,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=LORA_ALPHA, lora_dropout=0.05,
    bias="none", use_gradient_checkpointing="unsloth", random_state=42,
)

args = TrainingArguments(
    output_dir=RUN_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01, warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    logging_steps=20,
    report_to="none",
    optim="adamw_8bit", seed=42,
)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=split["train"], eval_dataset=split["test"],
    dataset_text_field="text", max_seq_length=2048,
    packing=True, args=args,
)

result = trainer.train()
eval_result = trainer.evaluate()

metrics = {{
    "run_name": RUN_NAME,
    "model_type": MODEL_TYPE,
    "lora_rank": LORA_RANK,
    "learning_rate": LEARNING_RATE,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "train_loss": result.metrics.get("train_loss"),
    "eval_loss": eval_result.get("eval_loss"),
    "perplexity": math.exp(eval_result.get("eval_loss", 0)),
    "runtime_min": result.metrics.get("train_runtime",0)/60,
}}

with open(f"{{RUN_DIR}}/metrics.json","w") as f:
    json.dump(metrics, f, indent=2)

print(f"DONE {{MODEL_TYPE}}: eval_loss={{metrics['eval_loss']:.4f}} PPL={{metrics['perplexity']:.2f}}")
'''

    py_path = f"{EXP_DIR}/scripts/{run_name}.py"
    with open(py_path, "w") as f:
        f.write(py_script)

    sbatch = f"""#!/bin/bash
#SBATCH --job-name=bvi_{i:03d}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time={TIME}
#SBATCH --output={EXP_DIR}/logs/{run_name}_%j.out
#SBATCH --error={EXP_DIR}/logs/{run_name}_%j.err

source {ENV_PATH}/bin/activate
python3 {py_path}
"""
    sbatch_path = f"{EXP_DIR}/scripts/{run_name}.sh"
    with open(sbatch_path, "w") as f:
        f.write(sbatch)

    jobs.append({"name": run_name, "sh": sbatch_path, "model_type": model_type})

submit_all = f"#!/bin/bash\necho 'Submitujem {len(jobs)} jobov...'\n"
for job in jobs:
    submit_all += f"sbatch {job['sh']}\n"
submit_all += "echo 'Hotovo!'\n"

submit_path = f"{EXP_DIR}/submit_all.sh"
with open(submit_path, "w") as f:
    f.write(submit_all)
os.chmod(submit_path, 0o755)

collector = f"""#!/usr/bin/env python3
import glob, json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

results = []
for f in glob.glob("{EXP_DIR}/*/metrics.json"):
    with open(f) as fh:
        results.append(json.load(fh))

if not results:
    print("Žiadne výsledky ešte.")
    exit()

df = pd.DataFrame(results)
print("\\n=== EXP 2: BASE vs INSTRUCT ===")

for model_type in ["base","instruct"]:
    sub = df[df["model_type"]==model_type]
    if len(sub):
        best = sub.nsmallest(1,"eval_loss").iloc[0]
        avg = sub["eval_loss"].mean()
        print(f"  {{model_type.upper()}}: best_eval_loss={{best['eval_loss']:.4f}}, avg={{avg:.4f}}, best_config=r{{best['lora_rank']}}_lr{{best['learning_rate']:.0e}}")

# Boxplot
fig, ax = plt.subplots(figsize=(8,5))
data = [df[df["model_type"]=="base"]["eval_loss"].dropna(),
        df[df["model_type"]=="instruct"]["eval_loss"].dropna()]
ax.boxplot(data, labels=["Base (llama-3-8b)", "Instruct (llama-3.1-8B)"])
ax.set_ylabel("Eval Loss")
ax.set_title("Base vs Instruct – Distribution of Eval Loss")
ax.grid(True, axis="y")
plt.tight_layout()
plt.savefig("{EXP_DIR}/base_vs_instruct.png", dpi=150)
print(f"\\nGraf: {EXP_DIR}/base_vs_instruct.png")
df.to_csv("{EXP_DIR}/results.csv", index=False)
"""

with open(f"{EXP_DIR}/collect_results.py", "w") as f:
    f.write(collector)

print(f"✅ Jobov: {len(jobs)}")
print(f"Est. GPU hodín: ~{len(jobs)*4*4:,}")
print(f"Submit: bash {submit_path}")
