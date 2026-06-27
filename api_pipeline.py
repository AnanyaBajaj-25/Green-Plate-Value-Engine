import os
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")


def fetch_local_vegetarian_data(location_string: str) -> pd.DataFrame:
    """
    Queries Google Places Text Search API to pull local vegetarian establishments.
    Flattens the nested JSON response into a clean, flat Pandas DataFrame.
    """
    if not API_KEY:
        raise ValueError("Missing GOOGLE_PLACES_API_KEY. Please verify your .env file.")

    url = "https://places.googleapis.com/v1/places:searchText"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": (
            "places.id,"
            "places.displayName,"
            "places.rating,"
            "places.userRatingCount,"
            "places.priceLevel,"
            "places.servesVegetarianFood,"
            "places.location,"
            "places.formattedAddress"
        ),
    }

    json_data = {
        "textQuery": f"best vegetarian restaurants in {location_string}",
        "languageCode": "en",
        "maxResultCount": 20,
    }

    try:
        response = requests.post(url, headers=headers, json=json_data, timeout=5.0)
        if response.status_code != 200:
            print(f"Google API Error Status {response.status_code}: {response.text}")
            return pd.DataFrame()

        places_data = response.json().get("places", [])

        if not places_data:
            print(f"No vegetarian spots returned for location: {location_string}")
            return pd.DataFrame()

        flattened_records = []
        for place in places_data:
            record = {
                "place_id": place.get("id", ""),
                "name": place.get("displayName", {}).get("text", "Unknown Restaurant"),
                "rating": place.get("rating", np.nan),
                "review_count": place.get("userRatingCount", 0),
                "price_level_raw": place.get("priceLevel", "PRICE_LEVEL_MODERATE"),
                "vegetarian_served": place.get("servesVegetarianFood", False),
                "lat": place.get("location", {}).get("latitude"),
                "lon": place.get("location", {}).get("longitude"),
                "address": place.get("formattedAddress", ""),
            }
            flattened_records.append(record)

        return pd.DataFrame(flattened_records)

    except requests.exceptions.Timeout:
        print("Timeout Error: Google's server took too long to respond.")
        return pd.DataFrame()
    except requests.exceptions.ConnectionError:
        print("Connection Error: network interface unreachable.")
        return pd.DataFrame()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Protocol Error: {http_err}")
        return pd.DataFrame()


def fetch_place_text_signals(place_id: str) -> dict:
    """
    Fetches the free-tier text signals available from the Place Details endpoint:
      - editorialSummary  : a short editorial blurb written by Google
      - reviewSummary     : a Gemini-generated synthesis of all user reviews

    NOTE: Individual user 'reviews' require the Places API Pro SKU (paid tier).
    These two fields are available on the standard tier and contain rich
    vegetarian-relevant language suitable for keyword sentiment scoring.

    Returns a dict with keys 'editorial' and 'review_summary' (both str, may be empty).
    """
    if not API_KEY:
        return {"editorial": "", "review_summary": ""}

    url = f"https://places.googleapis.com/v1/places/{place_id}"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "editorialSummary,reviewSummary",
    }

    try:
        response = requests.get(url, headers=headers, timeout=5.0)
        if response.status_code != 200:
            print(f"Details API Error for {place_id}: {response.status_code}")
            return {"editorial": "", "review_summary": ""}

        body = response.json()
        editorial = body.get("editorialSummary", {}).get("text", "")
        review_summary = body.get("reviewSummary", {}).get("text", {}).get("text", "")
        return {"editorial": editorial, "review_summary": review_summary}

    except requests.exceptions.Timeout:
        print(f"Timeout fetching text signals for {place_id}")
        return {"editorial": "", "review_summary": ""}
    except Exception as e:
        print(f"Error fetching text signals for {place_id}: {e}")
        return {"editorial": "", "review_summary": ""}


if __name__ == "__main__":
    from ml_model import (
        process_restaurant_analytics,
        get_neighborhood_insights,
        calculate_sentiment_boost_from_text,
    )

    print("Executing local test pipeline...")

    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": (
            "places.id,"
            "places.displayName,"
            "places.rating,"
            "places.userRatingCount,"
            "places.priceLevel,"
            "places.servesVegetarianFood"
        ),
    }
    json_data = {
        "textQuery": "best vegetarian restaurants in La Jolla, San Diego",
        "languageCode": "en",
        "maxResultCount": 5,
    }

    try:
        response = requests.post(url, headers=headers, json=json_data, timeout=5.0)
        response.raise_for_status()
        places_data = response.json().get("places", [])

        flattened_records = []
        print("\nFetching free-tier text signals and running NLP keyword mining...")

        for place in places_data:
            name = place.get("displayName", {}).get("text", "Unknown Restaurant")
            base_rating = place.get("rating", 0.0)
            place_id = place.get("id", "")

            print(f"\nFetching text signals for: {name}")
            signals = fetch_place_text_signals(place_id)

            # Combine both text fields into a single string for scoring
            combined_text = f"{signals['editorial']} {signals['review_summary']}".strip()

            boosted_score, match_count, source = calculate_sentiment_boost_from_text(
                combined_text, base_rating, restaurant_name=name
            )

            record = {
                "name": name,
                "rating": base_rating,
                "review_count": place.get("userRatingCount", 0),
                "price_level_raw": place.get("priceLevel", "PRICE_LEVEL_MODERATE"),
                "value_score": boosted_score,
                "text_source": source,
                "keyword_matches": match_count,
            }
            flattened_records.append(record)
            print(
                f"{name}: {match_count} keyword matches from {source} "
                f"(Base: {base_rating} → Boosted: {boosted_score})"
            )

        test_df = pd.DataFrame(flattened_records)
        test_df = process_restaurant_analytics(test_df)
        insights = get_neighborhood_insights(test_df)
        print(f"\nNeighborhood Insights: {insights}")

    except requests.exceptions.Timeout:
        print("Timeout Error: request took too long.")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error from Places API: {e}")
    except requests.exceptions.ConnectionError:
        print("Connection Error: could not reach the API.")
    except ValueError as e:
        print(f"Data error: {e}")