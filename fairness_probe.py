# src/fairness_probe.py
print("Running Fairness probe (CrowS-Pairs via CSV)…")

import io
import sys
import pandas as pd
import requests
from typing import List, Tuple
from src.setup_utils import (
    load_gpt2,
    encode_texts_hidden_states,
    train_linear_probes,
    class_separation,
    save_json,
)

RAW_URLS = [
    # Primary: canonical repo raw CSV (anonymized)
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv",
    # Mirror path (if structure changes)
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/main/data/crows_pairs_anonymized.csv",
]

def fetch_crows_pairs_csv() -> pd.DataFrame:
    last_err = None
    for url in RAW_URLS:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            buf = io.StringIO(r.text)
            df = pd.read_csv(buf)
            print(f"[Fairness] Loaded CrowS-Pairs CSV from {url} with shape {df.shape}.")
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to fetch CrowS-Pairs CSV from known URLs. Last error: {last_err}")

def extract_texts_labels(df: pd.DataFrame, max_examples: int = 5000) -> Tuple[List[str], List[int]]:
    """
    Map:
      sent_more  (stereotype)      -> 0 (unfair)
      sent_less  (anti-stereotype) -> 1 (fair)
    Flattens each pair into two examples.
    """
    # Normalize column names just in case
    cols = {c.lower(): c for c in df.columns}
    more_col = cols.get("sent_more") or cols.get("sentence_more")
    less_col = cols.get("sent_less") or cols.get("sentence_less")
    if more_col is None or less_col is None:
        raise ValueError(f"CSV missing required columns. Found: {list(df.columns)}")

    texts, labels = [], []
    for _, row in df.iterrows():
        s = str(row[more_col]).strip() if pd.notnull(row[more_col]) else ""
        a = str(row[less_col]).strip() if pd.notnull(row[less_col]) else ""
        if s:
            texts.append(s); labels.append(0)  # stereotype -> unfair
        if a:
            texts.append(a); labels.append(1)  # anti-stereotype -> fair
        if len(texts) >= max_examples:
            break
    print(f"[Fairness] Prepared {len(texts)} text examples.")
    return texts, labels

if __name__ == "__main__":
    # 0) Soft dependency check
    try:
        import pandas  # noqa: F401
        import requests  # noqa: F401
    except Exception:
        print("Missing dependencies. Run: pip install pandas requests", file=sys.stderr)
        sys.exit(1)

    # 1) Fetch CSV
    df = fetch_crows_pairs_csv()

    # 2) Build texts/labels
    texts, labels = extract_texts_labels(df, max_examples=3000)

    # 3) Load model
    tokenizer, model, device = load_gpt2("gpt2-medium")

    # 4) Encode representations
    reps = encode_texts_hidden_states(texts, tokenizer, model, device, max_len=64, batch_size=16)

    # 5) Linear probe accuracies
    accs = train_linear_probes(reps, labels, test_size=0.2)

    # 6) Activation separation
    sep = class_separation(reps, labels)

    # 7) Save results
    save_json("results/fairness.json", {"accs": accs, "separation": sep})
