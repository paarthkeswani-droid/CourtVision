"""Interactive CourtVision recruiter demo."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

st.set_page_config(page_title="CourtVision", page_icon="🏀", layout="wide")
st.markdown("""
<style>
  .stApp { background: #f5f2e9; color: #17231f; }
  [data-testid="stSidebar"] { background: #183f39; }
  [data-testid="stSidebar"] * { color: #f6f1e5 !important; }
  .hero { background: linear-gradient(115deg,#183f39,#285b51); color:white; padding:28px 32px; border-radius:14px; margin-bottom:22px; }
  .hero small { color:#e6bd56; letter-spacing:.16em; font-weight:700; }
  .hero h1 { margin:.25rem 0; font-size:2.2rem; }
  .hero p { color:#c6d4cf; margin:0; }
  [data-testid="stMetric"] { background:#fffdf8; border:1px solid #dedbd1; padding:16px; border-radius:10px; }
  .card { background:#fffdf8; border:1px solid #dedbd1; padding:20px; border-radius:10px; }
  .pill { display:inline-block; padding:4px 9px; border-radius:20px; background:#f8e7df; color:#d85b38; font-size:.75rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

required = [OUT / "prospect_rankings.csv", OUT / "model_metrics.json", OUT / "feature_importance.csv"]
if not all(path.exists() for path in required):
    st.error("Demo outputs are missing. Run `make demo` first.")
    st.stop()

rankings = pd.read_csv(required[0])
metrics = json.loads(required[1].read_text())
importance = pd.read_csv(required[2])
selected_model = metrics["selected_model"]
holdout = metrics["models"][selected_model]

st.sidebar.markdown("## 🏀 COURTVISION")
st.sidebar.caption("Prospect intelligence workspace")
st.sidebar.divider()
page = st.sidebar.radio("Workspace", ["Model overview", "Prospect explorer", "Feature intelligence"])
st.sidebar.divider()
st.sidebar.caption("DEMO MODE")
st.sidebar.info("All players and outcomes are synthetic. This demo shows the workflow, not real scouting conclusions.")

st.markdown("""<div class="hero"><small>RECRUITER DEMO • 2025 CLASS</small><h1>See the prospect behind the box score.</h1><p>Explainable player grades, honest holdout evaluation, and quantitative historical comparisons.</p></div>""", unsafe_allow_html=True)

if page == "Model overview":
    a,b,c,d = st.columns(4)
    a.metric("Selected model", selected_model.replace("_", " ").title())
    b.metric("Holdout MAE", f"{holdout['mae']:.2f}")
    c.metric("Holdout RMSE", f"{holdout['rmse']:.2f}")
    d.metric("Holdout R²", f"{holdout['r2']:.2f}")
    st.caption("Metrics are generated from the deterministic synthetic demo and are included to demonstrate evaluation discipline.")
    left,right = st.columns(2)
    with left:
        st.subheader("Holdout performance")
        st.image(str(OUT / "predicted_vs_actual.png"), width="stretch")
    with right:
        st.subheader("Top statistical profiles")
        st.image(str(OUT / "top_prospects.png"), width="stretch")

elif page == "Prospect explorer":
    names = rankings.sort_values("model_grade", ascending=False).player.tolist()
    chosen = st.selectbox("Select a prospect", names)
    player = rankings.loc[rankings.player == chosen].iloc[0]
    st.markdown(f"### {chosen} &nbsp; <span class='pill'>{player['class_year']}</span>", unsafe_allow_html=True)
    a,b,c,d,e = st.columns(5)
    a.metric("CourtVision grade", f"{player.model_grade:.1f}")
    b.metric("Projected value", f"{player.predicted_nba_value:.2f}")
    c.metric("Age", f"{player.age:.1f}")
    d.metric("Height", f"{player.height_in:.1f} in")
    e.metric("True shooting", f"{player.ts_pct:.1%}")
    left,right = st.columns([1.25,1])
    with left:
        st.subheader("Scouting profile")
        profile = pd.DataFrame({"Metric":["Points / 40","Assists / 40","Rebounds / 40","Steals / 40","Blocks / 40","AST / TOV"],"Value":[player.pts_per40,player.ast_per40,player.reb_per40,player.stl_per40,player.blk_per40,player.ast_tov]}).set_index("Metric")
        st.bar_chart(profile, color="#205a50", horizontal=True)
    with right:
        st.subheader("Context")
        st.markdown(f"""<div class="card"><b>{player.school}</b><br><small>{player.conference} • {int(player.season)} season</small><hr>Draft slot: <b>{'Undrafted' if pd.isna(player.draft_pick) else f'#{int(player.draft_pick)}'}</b><br>Three-point rate: <b>{player.three_rate:.1%}</b><br>Free-throw rate: <b>{player.ft_rate:.1%}</b></div>""", unsafe_allow_html=True)
        st.info("Use this profile to identify film questions—not to replace scouting context.")
    st.subheader("Rankings table")
    st.dataframe(rankings[["player","school","age","ts_pct","ast_tov","predicted_nba_value","model_grade"]].head(25), hide_index=True, width="stretch")

else:
    st.subheader("What drives the selected model?")
    st.write("Magnitude shows influence; direction is preserved for the linear model. The chart helps a scout challenge what the model is learning.")
    display = importance.copy().sort_values("importance")
    st.bar_chart(display.set_index("feature"), color="#205a50", horizontal=True)
    st.dataframe(importance.rename(columns={"feature":"Feature","importance":"Model importance / coefficient"}), hide_index=True, width="stretch")
    st.warning("Feature importance is descriptive, not causal. Role, competition, injuries, measurements, and film remain essential.")

st.divider()
st.caption("CourtVision • Educational portfolio project • Synthetic demo data")

