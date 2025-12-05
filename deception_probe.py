# src/deception_probe.py
print("Running Deception probe (TruthfulQA)…")

from datasets import load_dataset
from src.setup_utils import (
    load_gpt2,
    encode_texts_hidden_states,
    train_linear_probes,
    class_separation,
    save_json,
)

def load_truthfulqa(max_examples=2000):
    """
    Load TruthfulQA dataset from Hugging Face and collapse labels.
    We'll use the 'mc1' task: each Q has answers labeled truthful vs. not.
    """
    ds = load_dataset("truthful_qa", "generation")
    texts, labels = [], []

    for ex in ds["validation"]:  # TruthfulQA only has validation split
        q = ex["question"]
        # 'best_answer' is labeled truthful, 'incorrect_answers' are deceptive
        if ex.get("best_answer"):
            texts.append(q + " " + ex["best_answer"])
            labels.append(0)  # truthful
        for bad in ex.get("incorrect_answers", []):
            texts.append(q + " " + bad)
            labels.append(1)  # deceptive
        if max_examples and len(texts) >= max_examples:
            break

    print(f"[Deception] TruthfulQA loaded: {len(texts)} examples "
          f"(truthful=0: {labels.count(0)}, deceptive=1: {labels.count(1)})")
    return texts, labels

if __name__ == "__main__":
    texts, labels = load_truthfulqa(max_examples=2000)

    tokenizer, model, device = load_gpt2("gpt2-medium")
    reps = encode_texts_hidden_states(texts, tokenizer, model, device, max_len=64, batch_size=16)

    accs = train_linear_probes(reps, labels, test_size=0.2)
    sep = class_separation(reps, labels)

    save_json("results/deception.json", {"accs": accs, "separation": sep})
