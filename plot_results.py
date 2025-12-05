# src/plot_results.py
import json
import matplotlib.pyplot as plt

def plot_results(json_path: str, title: str, out_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Parse and sort by numeric layer index
    acc_items = sorted(((int(k), v) for k, v in data["accs"].items()), key=lambda x: x[0])
    sep_items = sorted(((int(k), v) for k, v in data["separation"].items()), key=lambda x: x[0])

    layers_acc, accs = zip(*acc_items)
    layers_sep, seps = zip(*sep_items)

    # Sanity print to terminal so we know what we're plotting
    print(f"[Plot] Using JSON: {json_path}")
    print(f"[Plot] Acc min/max: {min(accs):.3f} / {max(accs):.3f}")
    print(f"[Plot] Sep min/max: {min(seps):.6f} / {max(seps):.6f}")

    # If accuracy looks like ~0.50, warn (likely wrong file)
    if max(accs) < 0.65:
        print("[Plot][WARN] Accuracy curve looks near chance. "
              "Are you sure this is the TruthfulQA deception JSON?")

    plt.figure(figsize=(6.4, 4.8))
    ax = plt.gca()

    # --- BLUE accuracy (make it very visible) ---
    ax.plot(layers_acc, accs, color="tab:blue", marker="o", linewidth=2.5, zorder=5, label="Accuracy")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")

    # Make sure the y-limits are reasonable for accuracy (forces the blue line into view)
    ax.set_ylim(0.40, 1.00)

    # --- RED separation on twin axis ---
    ax2 = ax.twinx()
    ax2.plot(layers_sep, seps, color="tab:red", linestyle="--", marker="x", linewidth=1.8, zorder=3, label="Separation")
    ax2.set_ylabel("Separation Score", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # 🔧 Force scientific notation for separation axis (for consistency)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # Title & layout
    ax.set_title(title)

    # ✅ Add legends for both Accuracy and Separation
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[Plot] Saved -> {out_path}")

if __name__ == "__main__":
    # Change the paths and titles to generate each figure
    plot_results(
        "results/intentionality.json",
        "Intentionality Probing (GPT-2-Medium)",
        "results/intentionality_plot.png"
    )

    plot_results(
        "results/fairness.json",
        "Fairness Probing (CrowS-Pairs, GPT-2-Medium)",
        "results/fairness_plot.png"
    )

    plot_results(
        "results/deception.json",
        "Deception Probing (TruthfulQA, GPT-2-Medium)",
        "results/deception_plot.png"
    )




