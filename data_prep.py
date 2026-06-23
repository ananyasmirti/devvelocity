"""
data_prep.py
Loads the Stack Overflow Developer Survey 2024 CSV and engineers
features mapped to the SPACE framework dimensions.

DATASET DOWNLOAD:
  https://survey.stackoverflow.co/2024/
  → Download the ZIP → extract survey_results_public.csv
  → Place it in devvelocity-ai/data/survey_results_public.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ── Ordinal encodings for string columns ─────────────────────────────────────

JOB_SAT_MAP = {
    "Very satisfied": 5,
    "Slightly satisfied": 4,
    "Neither satisfied nor dissatisfied": 3,
    "Slightly dissatisfied": 2,
    "Very dissatisfied": 1,
}

TIME_SEARCHING_MAP = {
    "Less than 15 minutes a day": 5,  # very efficient
    "15-30 minutes a day": 4,
    "30-60 minutes a day": 3,
    "1-2 hours a day": 2,
    "Over 2 hours a day": 1,          # very inefficient
}

TIME_ANSWERING_MAP = {
    "Less than 15 minutes a day": 1,
    "15-30 minutes a day": 2,
    "30-60 minutes a day": 3,
    "1-2 hours a day": 4,
    "Over 2 hours a day": 5,          # experienced = helps others more
}

ORG_SIZE_MAP = {
    "Just me - I am a freelancer, sole proprietor, etc.": 1,
    "2 to 9 employees": 2,
    "10 to 19 employees": 3,
    "20 to 99 employees": 4,
    "100 to 499 employees": 5,
    "500 to 999 employees": 6,
    "1,000 to 4,999 employees": 7,
    "5,000 to 9,999 employees": 8,
    "10,000 or more employees": 9,
}

REMOTE_MAP = {
    "Remote": 2,
    "Hybrid (some remote, some in-person)": 1,
    "In-person": 0,
}


# ── Load ──────────────────────────────────────────────────────────────────────

def load_survey(csv_path: str) -> pd.DataFrame:
    """
    Load the SO Developer Survey CSV and filter to professional developers only.
    
    Args:
        csv_path: Path to survey_results_public.csv
    Returns:
        Filtered DataFrame
    """
    print(f"📂 Loading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"   Raw rows: {len(df):,}")

    # Keep only professional developers (MainBranch column)
    if "MainBranch" in df.columns:
        df = df[df["MainBranch"] == "I am a developer by profession"].copy()
        print(f"   After filtering to professional devs: {len(df):,}")

    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map raw SO Survey columns to SPACE framework dimensions.

    SPACE dimensions:
      S  - Satisfaction
      P  - Performance
      A  - Activity
      C  - Collaboration
      E  - Efficiency / Flow
    
    Returns:
        DataFrame of numeric features ready for LightGBM
    """
    features = pd.DataFrame(index=df.index)

    # ── S: Satisfaction ───────────────────────────────────────────────────────
    # Column: JobSat → numeric 0-10 scale
    if "JobSat" in df.columns:
        features["sat_job_satisfaction"] = pd.to_numeric(df["JobSat"], errors="coerce")

    # ── P: Performance ────────────────────────────────────────────────────────
    # Proxy: years of professional coding experience
    if "YearsCodePro" in df.columns:
        features["perf_years_coding_pro"] = (
            pd.to_numeric(
                df["YearsCodePro"].replace({
                    "Less than 1 year": 0,
                    "More than 50 years": 51,
                }),
                errors="coerce",
            ).clip(0, 40)
        )

    if "WorkExp" in df.columns:
        features["perf_work_experience"] = (
            pd.to_numeric(
                df["WorkExp"].replace({
                    "Less than 1 year": 0,
                    "More than 50 years": 51,
                }),
                errors="coerce",
            ).clip(0, 40)
        )

    # ── A: Activity ───────────────────────────────────────────────────────────
    # Count of distinct coding activities (hobby, freelance, open source, etc.)
    if "CodingActivities" in df.columns:
        features["act_coding_activity_count"] = (
            df["CodingActivities"]
            .fillna("")
            .str.split(";")
            .apply(lambda x: len([a for a in x if a.strip()]))
        )

    # ── C: Collaboration ──────────────────────────────────────────────────────
    if "ICorPM" in df.columns:
        features["collab_is_manager"] = (
            df["ICorPM"]
            .str.contains("people manager", case=False, na=False)
            .astype(int)
        )

    if "OrgSize" in df.columns:
        features["collab_org_size"] = df["OrgSize"].map(ORG_SIZE_MAP)

    # ── E: Efficiency / Flow ──────────────────────────────────────────────────
    # Time NOT wasted searching = efficiency (higher = less searching = better)
    if "TimeSearching" in df.columns:
        features["eff_search_efficiency"] = df["TimeSearching"].map(TIME_SEARCHING_MAP)

    # Time answering others = senior knowledge sharing
    if "TimeAnswering" in df.columns:
        features["eff_time_answering"] = df["TimeAnswering"].map(TIME_ANSWERING_MAP)

    # AI tool adoption = automation efficiency
    if "AIToolCurrently Using" in df.columns:
        features["eff_ai_tool_count"] = (
            df["AIToolCurrently Using"]
            .fillna("")
            .str.split(";")
            .apply(lambda x: len([t for t in x if t.strip()]))
        )

    # Remote work flexibility
    if "RemoteWork" in df.columns:
        features["eff_remote_flexibility"] = df["RemoteWork"].map(REMOTE_MAP)

    return features


# ── Target creation ───────────────────────────────────────────────────────────

def create_space_target(features: pd.DataFrame) -> pd.Series:
    """
    Build a composite SPACE efficiency score and bin it into 3 classes:
      0 = Low efficiency
      1 = Medium efficiency
      2 = High efficiency

    Weights reflect what the SPACE framework says matters most for developer flow.
    """
    score_cols = [c for c in [
        "sat_job_satisfaction",    # S — 30%
        "perf_years_coding_pro",   # P — 15%
        "act_coding_activity_count",  # A — 10%
        "eff_search_efficiency",   # E — 30%
        "eff_ai_tool_count",       # E — 15%
    ] if c in features.columns]

    if not score_cols:
        raise ValueError("No SPACE columns found. Check your CSV columns.")

    filled = features[score_cols].fillna(features[score_cols].median())
    scaler = MinMaxScaler()
    normed = pd.DataFrame(
        scaler.fit_transform(filled),
        columns=score_cols,
        index=features.index,
    )

    WEIGHTS = {
        "sat_job_satisfaction":      0.30,
        "perf_years_coding_pro":     0.15,
        "act_coding_activity_count": 0.10,
        "eff_search_efficiency":     0.30,
        "eff_ai_tool_count":         0.15,
    }

    composite = sum(
        normed[col] * w
        for col, w in WEIGHTS.items()
        if col in normed.columns
    )

    # Bin into 3 equal-width classes
    target = pd.cut(composite, bins=3, labels=[0, 1, 2])
    return target.astype(float)
