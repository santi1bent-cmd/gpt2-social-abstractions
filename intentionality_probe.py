# src/intentionality_probe.py
print("Running Intentionality probe…")

from datasets import load_dataset
from src.setup_utils import (
    load_gpt2,
    encode_texts_hidden_states,
    train_linear_probes,
    class_separation,
    save_json,
)

def load_intentionality(max_examples=1000):
    """
    Dataset: tasksource/tomi-nli
    Label mapping:
    entailment -> 1 (intentional)
    neutral/contradiction -> 0 (not intentional)
    """
    ds = load_dataset("tasksource/tomi-nli", split="train")
    texts, labels = [], []
    for ex in ds:
        prem, hyp = ex.get("premise"), ex.get("hypothesis")
        if not prem or not hyp:
            continue
        y = 1 if str(ex.get("label", "")).strip().lower() == "entailment" else 0
        texts.append(prem + " " + hyp)
        labels.append(y)
        if len(texts) >= max_examples:
            break
    return texts, labels

if __name__ == "__main__":
    # 1. Load model
    tokenizer, model, device = load_gpt2("gpt2-medium")

    # 2. Load dataset
    texts, labels = load_intentionality(max_examples=1000)

    # 3. Encode representations
    reps = encode_texts_hidden_states(texts, tokenizer, model, device, max_len=128, batch_size=8)

    # 4. Linear probe accuracies
    accs = train_linear_probes(reps, labels, test_size=0.2)

    # 5. Activation analysis
    sep = class_separation(reps, labels)

    # 6. Save results
    save_json("results/intentionality.json", {"accs": accs, "separation": sep})
