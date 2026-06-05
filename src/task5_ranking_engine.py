"""Task 5: Candidate Ranking Engine — similarity scoring against a JD"""
import pandas as pd
import numpy as np
import re, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_JD = {
    "title"      : "Senior Data Scientist",
    "required_keywords": ["python", "machine learning", "data science",
                           "projects", "certifications"],
    "required_category": "Data Science",
}

def text_similarity(resume_text: str, jd_keywords: list) -> float:
    text   = str(resume_text).lower()
    hits   = sum(1 for kw in jd_keywords if kw.lower() in text)
    return round(hits / max(len(jd_keywords), 1) * 100, 1)

def run(data_path="data/resume_intelligence_500.csv",
        artifacts_dir="artifacts",
        plots_dir="plots",
        jd=None):
    os.makedirs(plots_dir, exist_ok=True)
    if jd is None:
        jd = DEFAULT_JD
    df = pd.read_csv(data_path)

    scores = []
    for _, row in df.iterrows():
        sim   = text_similarity(row["resume_text"], jd["required_keywords"])
        bonus = 10 if str(row["category"]).strip() == jd["required_category"] else 0
        scores.append(min(sim + bonus, 100))

    df["similarity_score"] = scores
    top10 = df.nlargest(10, "similarity_score")

    print("=" * 60)
    print(f"TASK 5 — Candidate Ranking for: {jd['title']}")
    print("=" * 60)
    print(top10[["resume_id", "category", "similarity_score"]].to_string(index=False))

    # Save results
    top10.to_csv(f"{artifacts_dir}/top10_candidates.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = ["gold", "silver", "#cd7f32"] + ["steelblue"] * 7
    ax.bar(range(len(top10)), top10["similarity_score"],
           color=colors[:len(top10)], edgecolor="black")
    ax.set_xticks(range(len(top10)))
    ax.set_xticklabels([str(r) for r in top10["resume_id"].tolist()],
                       rotation=45, ha="right")
    ax.set_title(f"Top 10 Candidate Rankings — {jd['title']}")
    ax.set_ylabel("Similarity Score (%)"); ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/candidate_rankings.png"); plt.close()
    print(f"\nPlot saved → {plots_dir}/candidate_rankings.png")
    return top10

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    run()
