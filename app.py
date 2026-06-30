import streamlit as st
import pandas as pd
import pydeck as pdk

from api_pipeline import fetch_local_vegetarian_data, fetch_place_text_signals
from ml_model import (
    calculate_sentiment_boost_from_text,
    get_neighborhood_insights,
    process_restaurant_analytics,
)

PRICE_DISPLAY = {
    "PRICE_LEVEL_INEXPENSIVE": "$",
    "PRICE_LEVEL_MODERATE": "$$",
    "PRICE_LEVEL_EXPENSIVE": "$$$",
    "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
}


def run_pipeline(location: str, progress_bar=None) -> tuple[pd.DataFrame, dict]:
    df = fetch_local_vegetarian_data(location)
    if df.empty:
        return df, get_neighborhood_insights(df)

    editorials, review_summaries = [], []
    value_scores, keyword_matches, text_sources = [], [], []
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        signals = fetch_place_text_signals(row["place_id"])
        combined = f"{signals['editorial']} {signals['review_summary']}".strip()
        base_rating = float(row["rating"]) if pd.notna(row["rating"]) else 0.0
        boosted, match_count, source = calculate_sentiment_boost_from_text(
            combined,
            base_rating,
            restaurant_name=row["name"],
            random_seed=hash(row["place_id"]) % (2**32),
        )
        editorials.append(signals["editorial"])
        review_summaries.append(signals["review_summary"])
        value_scores.append(boosted)
        keyword_matches.append(match_count)
        text_sources.append(source)
        if progress_bar is not None:
            progress_bar.progress((idx + 1) / total)

    df = df.copy()
    df["editorial"] = editorials
    df["review_summary"] = review_summaries
    df["value_score"] = value_scores
    df["keyword_matches"] = keyword_matches
    df["text_source"] = text_sources
    df = process_restaurant_analytics(df)
    return df, get_neighborhood_insights(df)


def format_price(raw: str) -> str:
    return PRICE_DISPLAY.get(raw, "$$")


def _truncate_label(text: str, max_len: int = 32) -> tuple[str, str | None]:
    """Shorten long labels for display; return full text when truncated."""
    if len(text) <= max_len:
        return text, None
    return text[: max_len - 1].rstrip() + "…", text


def render_insight(label: str, value: str, note: str = "") -> None:
    with st.container(border=True):
        st.caption(label)
        st.markdown(f"###### {value}")
        if note:
            st.caption(note)


def _score_to_color(score: float) -> list[int]:
    """Green gradient: lighter for lower scores, deeper green for elite spots."""
    t = min(max(score / 5.0, 0), 1)
    return [int(210 * (1 - t)), int(90 + 165 * t), int(70 * (1 - t)), 210]


def _compute_map_zoom(lat_span: float, lon_span: float) -> float:
    span = max(lat_span, lon_span, 0.004)
    return float(min(15, max(10, 14 - span * 350)))


def build_restaurant_map(map_df: pd.DataFrame) -> pdk.Deck:
    plot_df = map_df.copy()
    plot_df["price_label"] = plot_df["price_level_raw"].map(format_price)
    plot_df["value_score_label"] = plot_df["value_score"].map(lambda s: f"{s:.2f}")
    plot_df["rating_label"] = plot_df["rating"].map(lambda r: f"{r:.1f}")
    plot_df["color"] = plot_df.apply(
        lambda row: [255, 140, 0, 220]
        if row["data_integrity"] == "Suspicious Outlier"
        else _score_to_color(row["value_score"]),
        axis=1,
    )
    plot_df["radius"] = plot_df["value_score"].map(lambda s: 90 + s * 35)

    center_lat = plot_df["lat"].mean()
    center_lon = plot_df["lon"].mean()
    lat_span = plot_df["lat"].max() - plot_df["lat"].min()
    lon_span = plot_df["lon"].max() - plot_df["lon"].min()

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=plot_df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        opacity=0.88,
        stroked=True,
        get_line_color=[255, 255, 255, 200],
        line_width_min_pixels=2,
    )

    return pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=_compute_map_zoom(lat_span, lon_span),
            pitch=0,
        ),
        tooltip={
            "html": (
                "<b>{name}</b><br/>"
                "Value score: {value_score_label}<br/>"
                "Google rating: {rating_label} ({review_count} reviews)<br/>"
                "Price: {price_label}<br/>"
                "{address}"
            ),
            "style": {
                "backgroundColor": "#1a472a",
                "color": "white",
                "fontSize": "13px",
                "padding": "8px",
            },
        },
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    )


st.set_page_config(
    page_title="Green Plate Value Engine",
    page_icon="🥗",
    layout="wide",
)

st.title("Green Plate Value Engine")
st.caption(
    "Discover vegetarian-friendly restaurants in any neighborhood, "
    "scored by Google ratings plus plant-based sentiment signals."
)

with st.form("search_form", clear_on_submit=False):
    col_input, col_btn = st.columns([4, 1])
    with col_input:
        location = st.text_input(
            "Location",
            placeholder="e.g. La Jolla, San Diego",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Search", use_container_width=True)

if submitted:
    if not location.strip():
        st.warning("Enter a neighborhood or city to search.")
    else:
        try:
            with st.spinner(f"Searching vegetarian spots in {location.strip()}…"):
                progress = st.progress(0, text="Fetching text signals and scoring value…")
                df, insights = run_pipeline(location.strip(), progress_bar=progress)
                progress.empty()
                st.session_state["results_df"] = df
                st.session_state["insights"] = insights
                st.session_state["searched_location"] = location.strip()
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")

df = st.session_state.get("results_df")
insights = st.session_state.get("insights")
searched_location = st.session_state.get("searched_location")

if df is not None:
    if df.empty:
        st.info(f"No vegetarian restaurants found for **{searched_location}**. Try a different area.")
    else:
        st.subheader(f"Results for {searched_location}")

        top_rated_display, top_rated_full = _truncate_label(insights["top_rated"])

        m1, m2, m3 = st.columns(3)
        with m1:
            render_insight(
                "Elite Density",
                insights["density_pct"],
                "% with value score ≥ 4.5",
            )
        with m2:
            render_insight(
                "Typical Cost",
                insights["avg_cost"],
                "Most common price tier in this area",
            )
        with m3:
            top_note = top_rated_full if top_rated_full else "Highest value score in this area"
            render_insight("Top Rated", top_rated_display, top_note)

        with st.sidebar:
            st.header("Filters")
            sort_by = st.selectbox(
                "Sort by",
                ["Value Score", "Google Rating", "Review Count", "Price"],
            )
            integrity_filter = st.selectbox(
                "Data integrity",
                ["All", "Verified Profile", "Suspicious Outlier"],
            )
            min_score = st.slider("Min value score", 0.0, 5.0, 0.0, 0.1)

        filtered = df.copy()
        if integrity_filter != "All":
            filtered = filtered[filtered["data_integrity"] == integrity_filter]
        filtered = filtered[filtered["value_score"] >= min_score]

        sort_map = {
            "Value Score": ("value_score", False),
            "Google Rating": ("rating", False),
            "Review Count": ("review_count", False),
            "Price": ("price_numeric", True),
        }
        col, ascending = sort_map[sort_by]
        filtered = filtered.sort_values(col, ascending=ascending, na_position="last")

        map_df = filtered.dropna(subset=["lat", "lon"])
        if not map_df.empty:
            st.markdown("### Restaurant map")
            st.caption(
                "Hover or tap a pin for details. "
                "Green = higher value score · Orange = suspicious outlier · "
                "Larger pins = higher scores."
            )
            st.pydeck_chart(build_restaurant_map(map_df), use_container_width=True, height=520)

            legend_l, legend_m, legend_r = st.columns(3)
            legend_l.markdown("🟢 **Elite** (value ≥ 4.5)")
            legend_m.markdown("🟡 **Good** (value 3.5–4.5)")
            legend_r.markdown("🟠 **Outlier** (flagged by model)")

        st.markdown("### Restaurant list")
        st.markdown(f"**{len(filtered)}** restaurant{'s' if len(filtered) != 1 else ''}")

        for _, row in filtered.iterrows():
            price = format_price(row["price_level_raw"])
            veg_label = "Yes" if row["vegetarian_served"] else "No / Unknown"
            boost = round(row["value_score"] - row["rating"], 2)
            integrity_icon = "✓" if row["data_integrity"] == "Verified Profile" else "⚠"
            source_label = (
                "Google text signals"
                if row["text_source"] == "api_text_signals"
                else "Synthetic fallback"
            )

            with st.expander(
                f"**{row['name']}** — Value {row['value_score']:.2f} · {price} · {integrity_icon}",
                expanded=False,
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Value Score", f"{row['value_score']:.2f}", f"+{boost:.2f}" if boost > 0 else None)
                c2.metric("Google Rating", f"{row['rating']:.1f}", f"{int(row['review_count'])} reviews")
                c3.metric("Keyword Matches", int(row["keyword_matches"]))
                c4.metric("Veg Menu", veg_label)

                st.write(row["address"])
                st.caption(
                    f"{row['data_integrity']} · Scored from {source_label} · "
                    f"{int(row['keyword_matches'])} plant-based keyword hits"
                )

                if row["editorial"]:
                    st.markdown("**Editorial summary**")
                    st.write(row["editorial"])
                if row["review_summary"]:
                    st.markdown("**Review summary**")
                    st.write(row["review_summary"])
                if not row["editorial"] and not row["review_summary"]:
                    st.caption("No Google text signals available for this place.")

        with st.expander("Raw data table"):
            display_cols = [
                "name",
                "address",
                "rating",
                "value_score",
                "review_count",
                "price_level_raw",
                "vegetarian_served",
                "keyword_matches",
                "text_source",
                "data_integrity",
            ]
            st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

else:
    st.info("Search a location above to see vegetarian restaurant value scores.")
