# src/setup_utils.py
# Shared utilities for concept probing

import random, numpy as np, torch
from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def load_gpt2(model_name: str = "gpt2-medium"):
    """
    Load GPT-2 and tokenizer. Return (tokenizer, model, device).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

@torch.no_grad()
def encode_texts_hidden_states(texts: List[str], tokenizer, model, device,
                            max_len: int = 128, batch_size: int = 8) -> np.ndarray:
    """
    Return numpy array [N, L, H] where L = number of layers, H = hidden size.
    Each example is mean-pooled across tokens per layer.
    """
    all_chunks = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        inputs = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states[1:]   # skip embedding layer
        pooled = [h.mean(dim=1) for h in hs]  # [B,H] for each layer
        stacked = torch.stack(pooled, dim=1)  # [B,L,H]
        all_chunks.append(stacked.cpu().numpy())
    return np.concatenate(all_chunks, axis=0)

def train_linear_probes(reps, labels, test_size=0.2) -> Dict[int, float]:
    """
    Train a logistic regression probe on each layer.
    reps: numpy [N,L,H], labels: list of 0/1.
    Returns {layer: accuracy}.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        reps, labels, test_size=test_size, random_state=SEED, stratify=labels
    )
    n_layers = reps.shape[1]
    accs: Dict[int, float] = {}
    for layer in range(n_layers):
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")
        clf.fit(X_train[:, layer, :], y_train)
        pred = clf.predict(X_test[:, layer, :])
        accs[layer] = accuracy_score(y_test, pred)
    best = max(accs, key=accs.get)
    print(f"[Linear Probe] Best layer={best} acc={accs[best]:.3f}")
    return accs

def class_separation(reps, labels):
    """
    Simple activation analysis: cosine separation between class means.
    Returns {layer: score}, higher means more separated.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    X = reps
    y = np.asarray(labels).astype(int)
    sep = {}
    for layer in range(X.shape[1]):
        Xl = X[:, layer, :]
        m0 = Xl[y == 0].mean(axis=0)
        m1 = Xl[y == 1].mean(axis=0)
        sim = float(cosine_similarity(m0.reshape(1, -1), m1.reshape(1, -1))[0, 0])
        sep[layer] = 1.0 - sim
    best = max(sep, key=sep.get)
    print(f"[Activation Separation] Best layer={best} score={sep[best]:.3f}")
    return sep

def save_json(path: str, payload: Dict):
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved results -> {path}")
