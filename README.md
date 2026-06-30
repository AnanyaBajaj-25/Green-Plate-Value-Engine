# 🥗 Green Plate Value Engine

Discover vegetarian-friendly restaurants in any neighborhood, scored by Google ratings plus plant-based sentiment signals from review and editorial text.

## Features

- **Location search** — Enter a neighborhood or city to find nearby vegetarian restaurants via the Google Places API
- **Value scoring** — Combines Google star ratings with keyword-based sentiment boosts from editorial summaries and AI-generated review summaries
- **Anomaly detection** — Flags suspicious outliers using Isolation Forest on rating, review count, value score, and price
- **Interactive map** — PyDeck scatter plot with color-coded pins (green = high value, orange = flagged outlier)
- **Neighborhood insights** — Elite density, typical cost tier, and top-rated spot for the searched area
- **Filterable results** — Sort and filter by value score, rating, reviews, price, and data integrity

## How it works

```
Location search
      │
      ▼
Google Places Text Search  ──►  Restaurant list (ratings, price, coords)
      │
      ▼
Place Details (per restaurant)  ──►  editorialSummary + reviewSummary
      │
      ▼
Keyword sentiment boost  ──►  value_score = rating + plant-based keyword boost
      │
      ▼
Isolation Forest  ──►  Verified Profile vs Suspicious Outlier
      │
      ▼
Streamlit UI  ──►  Map, insights, expandable restaurant cards
```

**Value score** starts from the Google rating and adds up to +0.3 based on plant-based keywords (`vegetarian`, `vegan`, `plant-based`, `veggie`, `tofu`, `gluten-free`, `organic`). When Google text signals are unavailable, a labeled synthetic fallback is used instead.

## Prerequisites

- Python 3.10+
- A [Google Places API key](https://developers.google.com/maps/documentation/places/web-service/get-api-key) with Places API (New) enabled

## Setup

1. Clone the repository:

```bash
git clone https://github.com/<your-org>/Green-Plate-Value-Engine.git
cd Green-Plate-Value-Engine
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:

```env
GOOGLE_PLACES_API_KEY=your_api_key_here
```

### Streamlit Community Cloud

`.env` is not deployed (and should stay gitignored). Add your key in the cloud app instead:

1. Open your app on [share.streamlit.io](https://share.streamlit.io)
2. Go to **App settings** → **Secrets**
3. Paste:

```toml
GOOGLE_PLACES_API_KEY = "your_api_key_here"
```

4. Save and reboot the app

## Usage

### Streamlit app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`), enter a location such as `La Jolla, San Diego`, and click **Search**.

### CLI pipeline test

Run the API pipeline and scoring logic directly from the command line:

```bash
python api_pipeline.py
```

This fetches a small sample of restaurants in La Jolla and prints keyword matches, boosted scores, and neighborhood insights.

## Project structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit UI — search, map, filters, and restaurant details |
| `api_pipeline.py` | Google Places API client (text search + place details) |
| `ml_model.py` | Value scoring, anomaly detection, and neighborhood insights |
| `requirements.txt` | Python dependencies |

## How the value score works

```
value_score = base_google_rating + min(keyword_match_count × 0.08, 0.3)
```

If a restaurant has no editorial or review-summary text available from the API (common for smaller or newly-listed places), the app generates a clearly-labeled synthetic fallback description so the keyword scorer always has something to work with. Every restaurant card shows whether its score came from real API text signals or the synthetic fallback (`text_source` field), so the scoring is fully transparent.

## Data integrity flag

An `IsolationForest` (contamination = 0.15) is fit on `[rating, review_count, value_score, price_numeric]` per search to flag restaurants whose combination of these features is statistically unusual relative to others in the same result set — e.g. a very high rating with almost no reviews. This is a per-search relative flag, not an absolute fraud signal.

## Limitations / notes

- The Places API **standard tier** doesn't expose individual user reviews — only the editorial summary and an AI-generated review synthesis. Full review text would require the paid Pro SKU.
- Sentiment scoring is keyword-based, not a trained NLP classifier — it's a deliberately interpretable heuristic, not a black box.
- Anomaly detection is computed independently per search (per neighborhood), so "Suspicious Outlier" is relative to that specific result set rather than a global standard.
- API responses are not cached, so repeated searches for the same location will re-query Google Places.

## Possible extensions

- Cache API responses (e.g. SQLite or Redis) to reduce repeated calls and cost
- Swap the keyword scorer for a fine-tuned sentiment classifier (e.g. DistilBERT) on the review summaries
- Add historical tracking to see how value scores change over time
- Deploy with Streamlit Community Cloud or Docker + a cloud host

## Author

Built as a portfolio project demonstrating API integration, lightweight NLP, unsupervised anomaly detection, and interactive geospatial visualization with Streamlit.

