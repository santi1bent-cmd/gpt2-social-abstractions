# src/plot_overlay.py
import json
import matplotlib.pyplot as plt

def load_accs(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    acc_items = sorted(((int(k), v) for k, v in data["accs"].items()), key=lambda x: x[0])
    layers, accs = zip(*acc_items)
    return layers, accs

if __name__ == "__main__":
    # Load results
    layers_int, accs_int = load_accs("results/intentionality.json")
    layers_fair, accs_fair = load_accs("results/fairness.json")
    layers_dec, accs_dec = load_accs("results/deception.json")

    # Plot
    plt.figure(figsize=(7,5))
    plt.plot(layers_int, accs_int, color="tab:blue", marker="o", label="Intentionality")
    plt.plot(layers_fair, accs_fair, color="tab:green", marker="s", label="Fairness")
    plt.plot(layers_dec, accs_dec, color="tab:red", marker="^", label="Deception")

    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)  # chance line
    plt.xlabel("Layer")
    plt.ylabel("Linear Probe Accuracy")
    plt.title("Comparative Probe Accuracy Across Concepts (GPT-2-Medium)")
    plt.legend()
    plt.ylim(0.45, 0.85)
    plt.tight_layout()
    plt.savefig("results/comparative_overlay.png", dpi=200)
    print("Saved -> results/comparative_overlay.png")
