"""Task 1: Resume Analytics"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
import re, os

def run(data_path="data/resume_intelligence_500.csv", plots_dir="plots"):
    os.makedirs(plots_dir, exist_ok=True)
    df = pd.read_csv(data_path)

    print("=" * 55)
    print("TASK 1 — Resume Analytics")
    print("=" * 55)
    print(f"Total resumes : {len(df)}")
    print(f"\nCategory distribution:\n{df['category'].value_counts().to_string()}")

    df["text_length"] = df["resume_text"].astype(str).apply(len)
    print(f"\nAvg resume length : {df['text_length'].mean():.1f} chars")

    # Common skills/words
    all_text = " ".join(df["resume_text"].astype(str)).lower()
    words    = re.findall(r'\b[a-z]{4,}\b', all_text)
    stop     = {"candidate","skills","relevant","completed","projects","certifications",
                "this","that","with","from","have","been","will","also","and","the","for"}
    words    = [w for w in words if w not in stop]
    top20    = Counter(words).most_common(20)
    print(f"\nTop 20 skill keywords:")
    for w, c in top20:
        print(f"  {w:22s}: {c}")

    # Plot 1: Category distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    df["category"].value_counts().plot(kind="bar", ax=ax,
                                       color="steelblue", edgecolor="black")
    ax.set_title("Resume Category Distribution")
    ax.set_xlabel("Category"); ax.set_ylabel("Count")
    plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(f"{plots_dir}/category_distribution.png"); plt.close()

    # Plot 2: Skill word frequency (word cloud alternative)
    wlabels, wcounts = zip(*top20)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(wlabels, wcounts, color="coral", edgecolor="black")
    ax.set_title("Top 20 Skill Keyword Frequencies")
    ax.set_xlabel("Count"); ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/skill_frequency.png"); plt.close()

    # Plot 3: Resume length distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["text_length"], bins=20, color="mediumseagreen", edgecolor="black")
    ax.set_title("Resume Length Distribution")
    ax.set_xlabel("Character Count"); ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/resume_length.png"); plt.close()

    print(f"\nPlots saved → {plots_dir}/")
    return df

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
