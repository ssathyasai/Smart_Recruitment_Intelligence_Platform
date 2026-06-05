"""
train.py — run ALL tasks for Recruitment Platform.
Usage:  python train.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

BASE      = os.path.dirname(__file__)
DATA      = os.path.join(BASE, "data",      "resume_intelligence_500.csv")
ARTIFACTS = os.path.join(BASE, "artifacts")
MODELS    = os.path.join(BASE, "models")
PLOTS     = os.path.join(BASE, "plots")

for d in [ARTIFACTS, MODELS, PLOTS]:
    os.makedirs(d, exist_ok=True)

from src.task1_resume_analytics  import run as analytics
from src.task2_text_engineering   import run as text_eng
from src.task3_attention_model    import run as attention
from src.task4_positional_encoding import run as pos_enc
from src.task5_ranking_engine      import run as ranking
from src.task6_explainability      import run as explain

if __name__ == "__main__":
    print("\n[1/6] Resume Analytics")
    analytics(DATA, PLOTS)

    print("\n[2/6] Text Engineering")
    text_eng(DATA, ARTIFACTS)

    print("\n[3/6] Attention Model")
    attention(ARTIFACTS, MODELS)

    print("\n[4/6] Positional Encoding")
    pos_enc(PLOTS)

    print("\n[5/6] Ranking Engine")
    ranking(DATA, ARTIFACTS, PLOTS)

    print("\n[6/6] Explainability")
    explain(DATA, ARTIFACTS, MODELS, PLOTS)

    print("\n✅  All tasks complete.")
    print("   Launch dashboard:  streamlit run app.py")
