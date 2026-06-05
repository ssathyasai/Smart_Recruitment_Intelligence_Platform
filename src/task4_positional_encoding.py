"""Task 4: Positional Encoding — prove resume order affects understanding"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))
    for pos in range(max_len):
        for i in range(0, d_model, 2):
            PE[pos, i]     = np.sin(pos / (10000 ** (2*i/d_model)))
            if i+1 < d_model:
                PE[pos, i+1] = np.cos(pos / (10000 ** (2*i/d_model)))
    return PE

def run(plots_dir="plots"):
    os.makedirs(plots_dir, exist_ok=True)
    MAX_LEN = 80
    D_MODEL = 64
    PE = positional_encoding(MAX_LEN, D_MODEL)

    print("=" * 60)
    print("TASK 4 — Positional Encoding (Resume Understanding)")
    print("=" * 60)

    resume_a = "candidate has skills relevant to web development completed projects and certifications"
    resume_b = "certifications and projects completed relevant skills web development candidate has"
    tA, tB   = resume_a.split(), resume_b.split()

    print(f"\nResume A: {resume_a}")
    print(f"Resume B: {resume_b}\n")
    print(f"PROOF — same words, different order → different PE vectors:\n")
    print(f"{'Word':<16} {'A-pos':>5}  {'A-PE[:3]':>30}   {'B-pos':>5}  {'B-PE[:3]'}")
    print("-" * 85)
    for word in ["candidate", "skills", "relevant", "development", "certifications"]:
        pa = tA.index(word) if word in tA else None
        pb = tB.index(word) if word in tB else None
        pe_a = PE[pa, :3].round(3) if pa is not None else "N/A"
        pe_b = PE[pb, :3].round(3) if pb is not None else "N/A"
        print(f"  {word:<16} {str(pa):>5}  {str(pe_a):>30}   {str(pb):>5}  {pe_b}")

    print("\n→ 'candidate' at pos 0 (Resume A) vs pos 8 (Resume B) = DIFFERENT embedding.")
    print("→ Model understands whether skills are listed FIRST or LAST in a resume.")

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(PE, aspect='auto', cmap='viridis')
    plt.colorbar(im)
    ax.set_title("Resume Token Positional Encoding Heatmap")
    ax.set_xlabel("Encoding Dimension"); ax.set_ylabel("Token Position in Resume")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/resume_pe_heatmap.png"); plt.close()
    print(f"\nHeatmap saved → {plots_dir}/resume_pe_heatmap.png")
    return PE

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
