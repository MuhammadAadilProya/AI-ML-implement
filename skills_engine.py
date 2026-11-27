"""
Shared model logic for predict.py (terminal) and app.py (Streamlit).
Loads the v2 models saved by analysis.ipynb (Phase 8).
"""
import os
import re
import warnings

import joblib
import numpy as np
from scipy import sparse

warnings.filterwarnings("ignore")


class SkillsEngine:
    def __init__(self, models_dir="models"):
        files = ["cleaner.pkl", "level_model.pkl", "recommender.pkl"]
        missing = [f for f in files if not os.path.exists(os.path.join(models_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"models/ mein ye files nahi milin: {missing}. Pehle naya analysis.ipynb poora chalayein."
            )
        self.cleaner = joblib.load(os.path.join(models_dir, "cleaner.pkl"))
        self.level = joblib.load(os.path.join(models_dir, "level_model.pkl"))
        self.R = joblib.load(os.path.join(models_dir, "recommender.pkl"))

        self.vocab = self.R["vocab"]
        self.vocab_index = {s: i for i, s in enumerate(self.vocab)}
        self.generic_mask = self.R["generic_mask"]
        self.p_global = self.R["p_global"]

    # ---------- Wahi cleaning jo notebook mein thi ----------
    @staticmethod
    def key_of(x):
        return re.sub(r"[^a-z0-9+#]", "", x)

    def basic_clean(self, x):
        x = re.sub(r"\s+", " ", x.strip().lower())
        x = re.sub(self.cleaner["leading_re"], "", x)
        x = re.sub(self.cleaner["trailing_re"], "", x)
        return self.cleaner["manual"].get(x, x).strip()

    def to_canonical(self, skill):
        return self.cleaner["canonical_by_key"].get(self.key_of(self.basic_clean(skill)))

    def parse_skills(self, skills):
        """skills: comma-separated text ya list. Returns (known, unknown)."""
        if isinstance(skills, str):
            skills = skills.split(",")
        known, unknown = set(), []
        for raw in skills:
            if not raw or not raw.strip():
                continue
            c = self.to_canonical(raw)
            if c and c in self.vocab_index:
                known.add(c)
            else:
                unknown.append(raw.strip())
        return sorted(known), unknown

    @staticmethod
    def clean_title(t):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", t.lower())).strip()

    # ---------- Recommender ----------
    def recommend(self, title, k_tech=10, k_general=5, min_share=0.15):
        R = self.R
        q = R["title_vec"].transform([self.clean_title(title)])
        dist, idx = R["nn"].kneighbors(q, n_neighbors=R["k"])
        sim = np.clip(1 - dist[0], 0, 1)
        w = sim ** 2
        W = sparse.csr_matrix((w, (np.zeros(len(w), dtype=int), idx[0])), shape=(1, len(R["uniq_titles"])))
        den = float((W @ R["n_jobs"])[0])
        P = (W @ R["S"]).toarray()[0] / max(den, 1e-9)

        score = P * np.log1p(P / np.maximum(self.p_global, 1e-6))
        score[self.generic_mask] = -1
        tech_idx = [i for i in np.argsort(-score)[:k_tech] if P[i] >= min_share]
        tech_idx.sort(key=lambda i: -P[i])
        gen_idx = [i for i in np.argsort(-P) if self.generic_mask[i] and P[i] > 0][:k_general]
        match = float(sim.max())
        return {
            "technical": [(self.vocab[i], float(P[i])) for i in tech_idx],
            "general": [(self.vocab[i], float(P[i])) for i in gen_idx],
            "similar_titles": list(R["uniq_titles"][idx[0][:5]]),
            "match": match,
            "found": match >= 0.2 and len(tech_idx) > 0,
            "weak": match < 0.3,
        }

    # ---------- Job level ----------
    def predict_level(self, title, skills):
        known, unknown = self.parse_skills(skills)
        L = self.level
        X = sparse.hstack([L["mlb"].transform([known]), L["tfidf"].transform([self.clean_title(title)])]).tocsr()
        probs = L["clf"].predict_proba(X)[0]
        ranked = sorted(zip(L["clf"].classes_, probs), key=lambda x: -x[1])
        return [(c, float(p)) for c, p in ranked], known, unknown

    # ---------- Skill gap ----------
    def skill_gap(self, title, skills):
        mine, unknown = self.parse_skills(skills)
        mine = set(mine)
        r = self.recommend(title)
        tech = [s for s, _ in r["technical"]]
        have = [s for s in tech if s in mine]
        missing = [(s, p) for s, p in r["technical"] if s not in mine]
        return {
            "rec": r,
            "have": have,
            "missing": missing,
            "score": len(have) / len(tech) if tech else 0.0,
            "extra": sorted(s for s in mine if s not in tech),
            "unknown": unknown,
            "general_missing": [s for s, _ in r["general"] if s not in mine],
        }
