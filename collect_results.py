#!/usr/bin/env python3
import glob, json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

results = []
for f in glob.glob("/leonardo_scratch/large/userexternal/lmalicko/AIFAC_WORKING/sweep_v3/sweep_20260328_210408/*/metrics.json"):
    with open(f) as fh:
        results.append(json.load(fh))

if not results:
    print("Žiadne výsledky ešte.")
    exit()

df = pd.DataFrame(results)
print(f"\n=== SWEEP VÝSLEDKY ===")
print(f"Celkom runov: {len(df)}")

# Top 10 konfigurácií
print("\nTop 10 podľa eval_loss:")
top = df.nsmallest(10, "eval_loss")[["run_name","lora_rank","learning_rate","num_epochs","lora_dropout","seed","eval_loss","perplexity"]]
print(top.to_string(index=False))

# Mean ± std pre každú konfiguráciu (agregované cez seeds)
df_agg = df.groupby(["lora_rank","learning_rate","num_epochs","lora_dropout"]).agg(
    eval_loss_mean=("eval_loss","mean"),
    eval_loss_std=("eval_loss","std"),
    perplexity_mean=("perplexity","mean"),
    n_runs=("eval_loss","count")
).reset_index()

print("\nNajlepšie konfigurácie (mean ± std cez seeds):")
top_agg = df_agg.nsmallest(10, "eval_loss_mean")
print(top_agg.to_string(index=False))

# Graf: rank vs eval_loss
fig, axes = plt.subplots(1,3,figsize=(18,5))

for ax, param, title in zip(axes, ["lora_rank","learning_rate","num_epochs"], ["LoRA Rank","Learning Rate","Num Epochs"]):
    grouped = df.groupby(param)["eval_loss"].agg(["mean","std"]).reset_index()
    ax.bar([str(x) for x in grouped[param]], grouped["mean"], yerr=grouped["std"], capsize=5)
    ax.set_xlabel(title); ax.set_ylabel("Eval Loss (mean ± std)")
    ax.set_title(f"Effect of {title}"); ax.grid(True, axis="y")

plt.tight_layout()
plt.savefig("/leonardo_scratch/large/userexternal/lmalicko/AIFAC_WORKING/sweep_v3/sweep_20260328_210408/sweep_results.png", dpi=150)

df.to_csv("/leonardo_scratch/large/userexternal/lmalicko/AIFAC_WORKING/sweep_v3/sweep_20260328_210408/all_results.csv", index=False)
df_agg.to_csv("/leonardo_scratch/large/userexternal/lmalicko/AIFAC_WORKING/sweep_v3/sweep_20260328_210408/aggregated_results.csv", index=False)

best = df_agg.nsmallest(1,"eval_loss_mean").iloc[0]
print(f"\n🏆 NAJLEPŠIA KONFIGURÁCIA:")
print(f"   rank={best['lora_rank']}, lr={best['learning_rate']:.0e}, epochs={best['num_epochs']}, dropout={best['lora_dropout']}")
print(f"   eval_loss = {best['eval_loss_mean']:.4f} ± {best['eval_loss_std']:.4f}")
print(f"\nGraf: /leonardo_scratch/large/userexternal/lmalicko/AIFAC_WORKING/sweep_v3/sweep_20260328_210408/sweep_results.png")
