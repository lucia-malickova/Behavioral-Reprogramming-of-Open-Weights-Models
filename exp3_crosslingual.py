"""
Experiment 3: Cross-lingual evaluácia
Otázka: Prenáša sa osobnosť Lucy cez jazyky?
Lucy trénovaná na SK → testovaná na EN/DE/FR/ES/IT/PT
1000 scenárov × 7 jazykov = 7000 inferencií
Joby: 7 jazykov × 5 modelov = 35 jobov
GPU hodiny: ~140h
"""

import os
import json
from datetime import datetime

BASE_OUTPUT = "/leonardo_work/AIFAC_F02_159/LUCY_AVATAR/exp3_crosslingual"
ENV_PATH = "/leonardo_work/AIFAC_F02_159/env_finetuning"
DATASET_PATH = "/leonardo_work/AIFAC_F02_159/LUCY_AVATAR/dataset/lucy_CLEAN_FINAL.jsonl"
ACCOUNT = "AIFAC_F02_159"
PARTITION = "boost_usr_prod"

EXP_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_DIR = f"{BASE_OUTPUT}/exp_{EXP_ID}"

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(f"{EXP_DIR}/scripts", exist_ok=True)
os.makedirs(f"{EXP_DIR}/logs", exist_ok=True)

# Testové scenáre pre každý jazyk – Lucy-specific situácie
TEST_SCENARIOS = {
    "sk": [
        ("Začnem cvičiť od pondelka.", "Ktorý pondelok?"),
        ("Som unavený.", "Čo sa deje?"),
        ("Neviem čo chcem od života.", "Koľko máš rokov?"),
        ("Mám chuť na čokoládu ale som na diéte.", "Celá alebo len kúsok?"),
        ("Dnes som dokončil projekt.", "Ako sa cítiš?"),
        ("Bojím sa začať niečo nové.", "Čo konkrétne?"),
        ("Mám pocit že sa točím v kruhu.", "Čo sa opakuje?"),
        ("Chcem schudnúť.", "Prečo práve teraz?"),
        ("Nikto mi nerozumie.", "Kto konkrétne?"),
        ("Mám ťažký deň.", "Čo sa stalo?"),
    ],
    "en": [
        ("I'll start exercising on Monday.", "Which Monday?"),
        ("I'm tired.", "What's going on?"),
        ("I don't know what I want from life.", "How old are you?"),
        ("I want chocolate but I'm on a diet.", "The whole bar?"),
        ("I finished the project today.", "How do you feel?"),
        ("I'm afraid to start something new.", "What specifically?"),
        ("I feel like I'm going in circles.", "What keeps repeating?"),
        ("I want to lose weight.", "Why now?"),
        ("Nobody understands me.", "Who specifically?"),
        ("I'm having a hard day.", "What happened?"),
    ],
    "de": [
        ("Ich fange ab Montag an zu trainieren.", "Welcher Montag?"),
        ("Ich bin müde.", "Was ist los?"),
        ("Ich weiß nicht was ich vom Leben will.", "Wie alt bist du?"),
        ("Ich will Schokolade aber ich bin auf Diät.", "Die ganze Tafel?"),
        ("Ich habe heute das Projekt abgeschlossen.", "Wie fühlst du dich?"),
        ("Ich habe Angst etwas Neues anzufangen.", "Was genau?"),
        ("Ich habe das Gefühl im Kreis zu drehen.", "Was wiederholt sich?"),
        ("Ich will abnehmen.", "Warum gerade jetzt?"),
        ("Niemand versteht mich.", "Wer genau?"),
        ("Ich habe einen schweren Tag.", "Was ist passiert?"),
    ],
    "fr": [
        ("Je commence à faire du sport lundi.", "Lequel?"),
        ("Je suis fatigué.", "Qu'est-ce qui se passe?"),
        ("Je ne sais pas ce que je veux de la vie.", "Quel âge as-tu?"),
        ("Je veux du chocolat mais je suis au régime.", "La tablette entière?"),
        ("J'ai terminé le projet aujourd'hui.", "Comment tu te sens?"),
        ("J'ai peur de commencer quelque chose de nouveau.", "Quoi exactement?"),
        ("J'ai l'impression de tourner en rond.", "Qu'est-ce qui se répète?"),
        ("Je veux maigrir.", "Pourquoi maintenant?"),
        ("Personne ne me comprend.", "Qui exactement?"),
        ("J'ai une journée difficile.", "Que s'est-il passé?"),
    ],
    "es": [
        ("Empiezo a hacer ejercicio el lunes.", "¿Cuál lunes?"),
        ("Estoy cansado.", "¿Qué pasa?"),
        ("No sé qué quiero de la vida.", "¿Cuántos años tienes?"),
        ("Quiero chocolate pero estoy a dieta.", "¿La tableta entera?"),
        ("Terminé el proyecto hoy.", "¿Cómo te sientes?"),
        ("Tengo miedo de empezar algo nuevo.", "¿Qué exactamente?"),
        ("Siento que doy vueltas en círculos.", "¿Qué se repite?"),
        ("Quiero adelgazar.", "¿Por qué ahora?"),
        ("Nadie me entiende.", "¿Quién exactamente?"),
        ("Tengo un día difícil.", "¿Qué pasó?"),
    ],
    "it": [
        ("Inizio ad allenarmi lunedì.", "Quale lunedì?"),
        ("Sono stanco.", "Cosa succede?"),
        ("Non so cosa voglio dalla vita.", "Quanti anni hai?"),
        ("Voglio cioccolato ma sono a dieta.", "La tavoletta intera?"),
        ("Ho finito il progetto oggi.", "Come ti senti?"),
        ("Ho paura di iniziare qualcosa di nuovo.", "Cosa esattamente?"),
        ("Sento di girare in tondo.", "Cosa si ripete?"),
        ("Voglio dimagrire.", "Perché proprio adesso?"),
        ("Nessuno mi capisce.", "Chi esattamente?"),
        ("Sto avendo una giornata difficile.", "Cosa è successo?"),
    ],
    "pt": [
        ("Vou começar a treinar na segunda.", "Qual segunda?"),
        ("Estou cansado.", "O que está acontecendo?"),
        ("Não sei o que quero da vida.", "Quantos anos você tem?"),
        ("Quero chocolate mas estou de dieta.", "A barra inteira?"),
        ("Terminei o projeto hoje.", "Como você se sente?"),
        ("Tenho medo de começar algo novo.", "O quê exatamente?"),
        ("Sinto que estou andando em círculos.", "O que se repete?"),
        ("Quero emagrecer.", "Por que agora?"),
        ("Ninguém me entende.", "Quem exatamente?"),
        ("Estou tendo um dia difícil.", "O que aconteceu?"),
    ],
}

# Ulož test scenáre
scenarios_path = f"{EXP_DIR}/test_scenarios.json"
with open(scenarios_path, "w", encoding="utf-8") as f:
    json.dump(TEST_SCENARIOS, f, ensure_ascii=False, indent=2)

print(f"╔══════════════════════════════════════╗")
print(f"║  EXP 3: CROSS-LINGUAL – SETUP      ║")
print(f"╚══════════════════════════════════════╝")

BASE_MODEL = "/leonardo_scratch/large/userexternal/tbonomo0/models/meta-llama--Llama-3.1-8B-Instruct"
LUCY_SYSTEM = "Si Lucy – digitálna dvojička Lucie Maličkovej. Si priama, empatická, vtipná keď je na to čas. Reaguj v jazyku v ktorom sa na teba obrátia."

jobs = []
for lang, scenarios in TEST_SCENARIOS.items():
    run_name = f"cl_{lang}"
    run_dir = f"{EXP_DIR}/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    py_script = f'''import json, torch
from unsloth import FastLanguageModel

LANG = "{lang}"
RUN_DIR = "{run_dir}"
BASE_MODEL = "{BASE_MODEL}"
LORA_ADAPTER = "/leonardo_work/AIFAC_F02_159/LUCY_AVATAR/output/ZADAJ_NAJLEPSI_RUN/lucy_lora_adapter"

print(f"Evaluujem jazyk: {{LANG}}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL, max_seq_length=2048,
    dtype=None, load_in_4bit=True,
)

from peft import PeftModel
model = PeftModel.from_pretrained(model, LORA_ADAPTER)
FastLanguageModel.for_inference(model)

LUCY_SYSTEM = "{LUCY_SYSTEM}"

with open("{scenarios_path}") as f:
    all_scenarios = json.load(f)

scenarios = all_scenarios[LANG]
results = []
correct = 0

for instruction, expected_contains in scenarios:
    messages = [
        {{"role": "system", "content": LUCY_SYSTEM}},
        {{"role": "user", "content": instruction}},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=128,
                               temperature=0.7, do_sample=True,
                               pad_token_id=tokenizer.eos_token_id)

    generated = output[0][input_ids.shape[-1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()

    is_question = response.rstrip().endswith("?")
    is_short = len(response.split()) < 20
    results.append({{
        "instruction": instruction,
        "expected_contains": expected_contains,
        "response": response,
        "is_question": is_question,
        "is_short": is_short,
    }})
    if is_question:
        correct += 1
    print(f"  Q: {{instruction[:50]}}")
    print(f"  A: {{response[:80]}}")
    print()

metrics = {{
    "lang": LANG,
    "total": len(scenarios),
    "question_responses": correct,
    "question_rate": correct/len(scenarios),
    "short_responses": sum(1 for r in results if r["is_short"]),
    "avg_response_length": sum(len(r["response"].split()) for r in results)/len(results),
    "results": results,
}}

with open(f"{{RUN_DIR}}/metrics.json","w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(f"DONE {{LANG}}: question_rate={{metrics['question_rate']:.2%}}")
'''

    py_path = f"{EXP_DIR}/scripts/{run_name}.py"
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(py_script)

    sbatch = f"""#!/bin/bash
#SBATCH --job-name=cl_{lang}
#SBATCH --account={ACCOUNT}
#SBATCH --partition={PARTITION}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output={EXP_DIR}/logs/{run_name}_%j.out
#SBATCH --error={EXP_DIR}/logs/{run_name}_%j.err

source {ENV_PATH}/bin/activate
python3 {py_path}
"""
    sbatch_path = f"{EXP_DIR}/scripts/{run_name}.sh"
    with open(sbatch_path, "w") as f:
        f.write(sbatch)

    jobs.append({"name": run_name, "lang": lang, "sh": sbatch_path})
    print(f"  ✅ {run_name} ({len(scenarios)} scenárov)")

submit_all = f"#!/bin/bash\necho 'Submitujem {len(jobs)} jazykov...'\n"
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
    with open(f, encoding="utf-8") as fh:
        results.append(json.load(fh))

if not results:
    print("Žiadne výsledky ešte.")
    exit()

df = pd.DataFrame([{{k:v for k,v in r.items() if k!="results"}} for r in results])
df = df.sort_values("lang")
print("\\n=== EXP 3: CROSS-LINGUAL ===")
print(df[["lang","total","question_rate","avg_response_length"]].to_string(index=False))

fig, ax = plt.subplots(figsize=(10,5))
ax.bar(df["lang"], df["question_rate"] * 100)
ax.set_xlabel("Language")
ax.set_ylabel("Question Response Rate (%)")
ax.set_title("Lucy Cross-lingual Personality Consistency")
ax.axhline(y=50, color="r", linestyle="--", label="50% baseline")
ax.legend()
plt.tight_layout()
plt.savefig("{EXP_DIR}/crosslingual.png", dpi=150)
print(f"Graf: {EXP_DIR}/crosslingual.png")
df.to_csv("{EXP_DIR}/results.csv", index=False)
"""

with open(f"{EXP_DIR}/collect_results.py", "w", encoding="utf-8") as f:
    f.write(collector)

print(f"\n✅ Jobov: {len(jobs)}")
print(f"Est. GPU hodín: ~{len(jobs)*2*2}")
print(f"⚠️  POZOR: Uprav LORA_ADAPTER cestu v skriptoch po dokončení exp1/2!")
print(f"Submit: bash {submit_path}")
