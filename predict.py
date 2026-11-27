"""
LinkedIn Skills Explorer - Terminal Interface (English)
Run command: python predict.py
"""
from skills_engine import SkillsEngine

try:
    engine = SkillsEngine("models")
except FileNotFoundError as e:
    raise SystemExit(str(e))


def bar(p):
    return "█" * int(p * 20)


def not_found(r):
    if not r["found"]:
        print("\n⚠️  Job title not found in dataset. Please check spelling or try a standard title (e.g., Software Engineer).")
        return True
    if r["weak"]:
        print("\n⚠️  Low dataset coverage for this specific title; predictions may be less precise.")
    return False


def option_recommend():
    title = input("Enter Job Title (e.g., Data Scientist, Registered Nurse): ").strip()
    if not title:
        return
    r = engine.recommend(title)
    if not_found(r):
        return
    print(f"\nMatching Job Titles: {', '.join(r['similar_titles'][:4])}")
    print("\nTechnical & Field-Specific Skills (% of job postings requesting this skill):")
    for i, (s, p) in enumerate(r["technical"], 1):
        print(f"  {i:2d}. {s:30s} {p:5.0%} {bar(p)}")
    print("\nGeneral & Soft Skills:")
    for s, p in r["general"]:
        print(f"      {s:30s} {p:5.0%} {bar(p)}")


def option_level():
    title = input("Enter Job Title: ").strip()
    skills = input("Enter Skills (comma-separated, e.g., Python, SQL, AWS): ").strip()
    if not title:
        return
    probs, known, unknown = engine.predict_level(title, skills)
    best, best_p = probs[0]
    print(f"\nPredicted Level: {best} ({best_p:.0%} confidence)")
    for c, p in probs:
        print(f"  {c:15s} {p:5.0%} {bar(p)}")
    print(f"Skills utilized by model: {', '.join(known) if known else 'None'}")
    if unknown:
        print(f"Unrecognized skills (ignored): {', '.join(unknown)}")
    print("Note: Title keywords (e.g., Senior, Junior, Lead) carry higher predictive weight than skills.")


def option_gap():
    title = input("Enter Target Job Title (e.g., Machine Learning Engineer): ").strip()
    skills = input("Enter Your Current Skills (comma-separated): ").strip()
    if not title:
        return
    g = engine.skill_gap(title, skills)
    if not_found(g["rec"]):
        return
    total = len(g["have"]) + len(g["missing"])
    print(f"\nSkill Coverage for '{title}': You have {len(g['have'])} of {total} key technical skills ({g['score']:.0%})")
    print(f"  ✅ Skills You Possess: {', '.join(g['have']) if g['have'] else 'None'}")
    print("  📚 Recommended Skills to Acquire (ranked by demand):")
    for s, p in g["missing"]:
        print(f"      - {s:28s} Requested by {p:5.0%} of postings")
    if g["extra"]:
        print(f"  ➕ Additional Skills You Have: {', '.join(g['extra'])}")
    if g["unknown"]:
        print(f"  ❔ Unrecognized Skills: {', '.join(g['unknown'])}")
    if g["general_missing"]:
        print(f"  💬 Soft Skills to Highlight on Resume: {', '.join(g['general_missing'])}")


def main():
    options = {"1": option_recommend, "2": option_level, "3": option_gap}
    while True:
        print("\n" + "=" * 55)
        print("  LinkedIn Skills Explorer & ML Models")
        print("=" * 55)
        print("  1. Recommend Skills for a Job Title")
        print("  2. Predict Job Level (Associate / Mid-Senior)")
        print("  3. Skill Gap Analysis")
        print("  q. Exit")
        choice = input("Select an option: ").strip().lower()
        if choice == "q":
            print("Goodbye!")
            break
        if choice in options:
            options[choice]()
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 'q'.")


if __name__ == "__main__":
    main()