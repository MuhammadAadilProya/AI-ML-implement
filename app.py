"""
LinkedIn Skills Explorer - Streamlit Web Application
Run command: streamlit run app.py
"""
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

from skills_engine import SkillsEngine

st.set_page_config(
    page_title="LinkedIn Skills Explorer",
    page_icon="💼",
    layout="wide"
)

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a8a69f"
EXAMPLES = [
    "Full Stack Developer", "Data Scientist", "Machine Learning Engineer", 
    "Software Engineer", "Registered Nurse", "Accountant", 
    "Sales Manager", "Graphic Designer"
]


@st.cache_resource(show_spinner="Loading machine learning models...")
def load_engine():
    return SkillsEngine("models")


try:
    engine = load_engine()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


def share_chart(items, color, x_title="Share of Job Postings Requesting Skill"):
    """Renders a responsive horizontal bar chart with proper vertical spacing."""
    d = pd.DataFrame(items, columns=["item", "share"])
    chart_height = max(140, 38 * len(d) + 40)
    
    chart = (
        alt.Chart(d)
        .mark_bar(color=color, cornerRadiusEnd=4, size=20)
        .encode(
            x=alt.X("share:Q", title=x_title, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("item:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300, labelPadding=10)),
            tooltip=[
                alt.Tooltip("item:N", title="Item"), 
                alt.Tooltip("share:Q", title="Share / Probability", format=".0%")
            ],
        )
        .properties(height=chart_height)
    )
    return chart


def title_input(key, label="Job Title"):
    c1, c2 = st.columns([2, 1])
    example = c2.selectbox("Or select an example", ["-"] + EXAMPLES, key=f"{key}_ex")
    default = "" if example == "-" else example
    title = c1.text_input(
        label, 
        value=default, 
        key=f"{key}_title_{default}",
        placeholder="e.g., Data Scientist, Registered Nurse, Full Stack Developer"
    )
    return title.strip()


def skills_input(key, label="Your Current Skills"):
    picked = st.multiselect(
        label, 
        options=engine.vocab, 
        key=f"{key}_ms",
        placeholder="Search and select skills (e.g., Python, React, SQL)"
    )
    extra = st.text_input("Unlisted skills (comma-separated, optional):", key=f"{key}_extra")
    return picked + [s for s in extra.split(",") if s.strip()]


def show_title_status(r):
    if not r["found"]:
        st.warning("Job title not found in the dataset. Please check your spelling or try a standard title (e.g., Software Engineer).")
        return False
    if r["weak"]:
        st.info("Limited dataset coverage for this title; results may be less precise.")
    st.caption("Matching Job Titles in Dataset: " + ", ".join(r["similar_titles"][:5]))
    return True


# ---------- Sidebar ----------
with st.sidebar:
    st.header("💼 LinkedIn Skills Explorer")
    
    if os.path.exists("results/metrics.json"):
        with open("results/metrics.json") as f:
            m = json.load(f)
        st.subheader("Model Metrics")
        if "recommender_tfidf_p10_technical" in m:
            st.metric(
                "Recommender Precision@10",
                f"{m['recommender_tfidf_p10_technical']:.2f}",
                f"{m['recommender_tfidf_p10_technical'] - m.get('recommender_baseline_p10_technical', 0):+.2f} vs baseline"
            )
        if "job_level_model_f1_macro" in m:
            st.metric(
                "Job Level Macro F1",
                f"{m['job_level_model_f1_macro']:.2f}",
                f"{m['job_level_model_f1_macro'] - m.get('job_level_baseline_f1_macro', 0):+.2f} vs baseline"
            )
        if "rows_used" in m:
            st.caption(f"{m['rows_used']:,} Processed Job Postings | {m.get('vocab_size', len(engine.vocab)):,} Vocabulary Skills")

# ---------- Main Page ----------
st.title("💼 LinkedIn Skills & Career Explorer")
st.write("Explore in-demand skills for any career path, analyze skill gaps, and evaluate job level predictions.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Skill Recommender", 
    "🎯 Skill Gap Analysis", 
    "📈 Job Level Predictor", 
    "📊 Dataset Insights"
])

# ---------- Tab 1: Recommender ----------
with tab1:
    st.subheader("Recommended Skills for a Job Role")
    title = title_input("rec")
    if title:
        r = engine.recommend(title, k_tech=12)
        if show_title_status(r):
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown("**Technical & Field-Specific Skills**")
                st.altair_chart(share_chart(r["technical"], BLUE), use_container_width=True)
            with c2:
                st.markdown("**General & Soft Skills**")
                st.altair_chart(share_chart(r["general"], GRAY), use_container_width=True)
            st.caption("Percentage reflects the proportion of similar job postings requiring the skill.")

# ---------- Tab 2: Skill Gap ----------
with tab2:
    st.subheader("Identify Skill Gaps for Target Roles")
    title = title_input("gap", "Target Job Title")
    skills = skills_input("gap")
    if title:
        g = engine.skill_gap(title, skills)
        if show_title_status(g["rec"]):
            total = len(g["have"]) + len(g["missing"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Match Score", f"{g['score']:.0%}")
            c2.metric("Skills You Possess", f"{len(g['have'])} / {total}")
            c3.metric("Skills to Acquire", len(g["missing"]))
            st.progress(g["score"])

            left, right = st.columns(2)
            with left:
                st.markdown("**✅ Skills You Possess**")
                st.write(", ".join(g["have"]) if g["have"] else "None identified")
                if g["extra"]:
                    st.markdown("**➕ Additional Skills**")
                    st.write(", ".join(g["extra"]))
                if g["unknown"]:
                    st.markdown("**质疑 Unrecognized Skills**")
                    st.write(", ".join(g["unknown"]))
            with right:
                st.markdown("**📚 Recommended Skills to Acquire (Ranked by Demand)**")
                if g["missing"]:
                    st.altair_chart(share_chart(g["missing"], ORANGE), use_container_width=True)
                else:
                    st.success("You meet all top skill requirements for this role! 🎉")
            if g["general_missing"]:
                st.info("Consider highlighting these soft skills on your resume: " + ", ".join(g["general_missing"]))

# ---------- Tab 3: Job Level ----------
with tab3:
    st.subheader("Predict Job Seniority Level")
    title = title_input("lvl")
    skills = skills_input("lvl", "Job Skills")
    if title:
        probs, known, unknown = engine.predict_level(title, skills)
        best, best_p = probs[0]
        
        st.metric("Predicted Seniority Level", best, f"{best_p:.0%} Confidence", delta_color="off")
        st.altair_chart(
            share_chart(probs, BLUE, x_title="Probability"), 
            use_container_width=True
        )
        st.caption("Skills evaluated by model: " + (", ".join(known) if known else "None"))
        if unknown:
            st.caption("Unrecognized skills ignored: " + ", ".join(unknown))
        st.info("Note: Job title keywords (e.g., Senior, Junior, Lead) carry higher predictive weight than skill lists. Neutral titles without skills output baseline class probabilities.")

# ---------- Tab 4: Insights ----------
with tab4:
    st.subheader("Exploratory Data Analysis & Visualizations")
    charts = [
        ("01b_top_technical_skills.png", "Top Technical & Field Skills"),
        ("03_tech_skills.png", "Top Skills in CS & Tech Roles"),
        ("05_recommendations.png", "Skill Recommender Sample Outputs"),
        ("06_model_comparison.png", "Model Performance Evaluation"),
        ("07_level_signals.png", "Feature Importance for Job Levels"),
        ("02_job_levels.png", "Distribution of Job Postings by Level"),
    ]
    found = [(f, c) for f, c in charts if os.path.exists(os.path.join("images", f))]
    if not found:
        st.info("No generated charts found in the images/ directory. Run analysis.ipynb first.")
    for f, caption in found:
        st.markdown(f"**{caption}**")
        st.image(os.path.join("images", f))