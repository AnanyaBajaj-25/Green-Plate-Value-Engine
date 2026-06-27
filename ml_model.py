import random
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def process_restaurant_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full data pipeline: cleans data, engineers features, and applies
    Isolation Forest anomaly detection.

    Requires 'value_score' to already exist on the DataFrame — populate it
    by calling calculate_sentiment_boost_from_text() for each row first.
    """
    if df.empty:
        return df

    # =========================================================================
    # STEP 1: DATA CLEANING & CATEGORICAL MAPPING
    # =========================================================================
    price_map = {
        "PRICE_LEVEL_INEXPENSIVE": 1,
        "PRICE_LEVEL_MODERATE": 2,
        "PRICE_LEVEL_EXPENSIVE": 3,
        "PRICE_LEVEL_VERY_EXPENSIVE": 4,
    }
    df["price_numeric"] = df["price_level_raw"].map(price_map).fillna(2)
    df["rating"] = df["rating"].fillna(0.0)
    df["review_count"] = df["review_count"].fillna(0).astype(int)

    # =========================================================================
    # STEP 2: MACHINE LEARNING (ISOLATION FOREST ANOMALY DETECTION)
    # =========================================================================
    required = ["rating", "review_count", "value_score", "price_numeric"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"process_restaurant_analytics() missing required column(s): {missing}. "
            "Run calculate_sentiment_boost_from_text() for each row first."
        )

    X = df[required]
    clf = IsolationForest(contamination=0.15, random_state=42)
    df["anomaly_code"] = clf.fit_predict(X)
    df["data_integrity"] = np.where(
        df["anomaly_code"] == -1, "⚠️ Suspicious Outlier", "✅ Verified Profile"
    )

    return df


def get_neighborhood_insights(df: pd.DataFrame) -> dict:
    """
    Aggregates macro-level metrics for the top dashboard cards.
    """
    if df.empty:
        return {"density_pct": "0%", "avg_cost": "N/A", "top_rated": "N/A"}

    total_spots = len(df)
    elite_spots = len(df[df["value_score"] >= 4.5])
    density_pct = f"{int((elite_spots / total_spots) * 100)}%" if total_spots > 0 else "0%"

    mode_price = df["price_level_raw"].mode()
    price_map = {
        "PRICE_LEVEL_INEXPENSIVE": "$",
        "PRICE_LEVEL_MODERATE": "$$",
        "PRICE_LEVEL_EXPENSIVE": "$$$",
        "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
    }
    avg_cost = price_map.get(mode_price[0], "$$") if not mode_price.empty else "$$"

    # idxmax() is explicit — doesn't rely on the DataFrame being pre-sorted
    top_rated = (
        df.loc[df["value_score"].idxmax(), "name"]
        if "name" in df.columns
        else "Unknown"
    )

    return {"density_pct": density_pct, "avg_cost": avg_cost, "top_rated": top_rated}


def _generate_fallback_text(base_score: float, restaurant_name: str) -> str:
    """
    Generates a single realistic text blob when no API text signals are available.
    Kept separate so calculate_sentiment_boost_from_text() stays focused on scoring.
    """
    positive = [
        f"Amazing plant-based options at {restaurant_name}! Highly recommend the vegan dishes.",
        "Great atmosphere and a wonderful selection of vegetarian entrees. Fresh tofu and veggie options.",
        f"Best plant-based meal I've had. {restaurant_name} is a hidden gem for vegetarians and vegans.",
    ]
    neutral = [
        f"Decent spot. {restaurant_name} has some vegetarian items but is mostly a standard menu.",
        "Nice environment, average prices. Limited vegan options but the veggie burger is okay.",
    ]
    pool = positive if base_score >= 4.5 else neutral
    return " ".join(random.sample(pool, k=min(2, len(pool))))


def calculate_sentiment_boost_from_text(
    text: str,
    base_score: float,
    restaurant_name: str = "",
    random_seed: int | None = None,
) -> tuple[float, int, str]:
    """
    Scans a combined text string (editorialSummary + reviewSummary) for
    vegetarian-relevant keywords and returns a scored boost.

    The Places API standard tier provides two free text fields per place:
      - editorialSummary  : a short blurb from Google editors
      - reviewSummary     : a Gemini-generated synthesis of all user reviews
    Pass both concatenated as `text`. If both are empty, a clearly-labelled
    synthetic fallback is generated instead.

    Returns:
        (boosted_score, match_count, source_label)
        source_label is one of: 'api_text_signals' | 'synthetic_fallback'
    """
    if random_seed is not None:
        random.seed(random_seed)

    source = "api_text_signals"
    if not text or not text.strip():
        print(
            f"  ℹ️  No text signals from API for '{restaurant_name}' — using synthetic fallback."
        )
        text = _generate_fallback_text(base_score, restaurant_name)
        source = "synthetic_fallback"

    keywords = ["vegetarian", "vegan", "plant-based", "veggie", "tofu", "gluten-free", "organic"]
    text_lower = text.lower()
    match_count = sum(text_lower.count(kw) for kw in keywords)

    boost = min(match_count * 0.08, 0.3)
    return round(base_score + boost, 2), match_count, source