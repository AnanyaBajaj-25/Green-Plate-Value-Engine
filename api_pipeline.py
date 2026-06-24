import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environmental variables from your hidden .env file
load_dotenv()
API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

def fetch_local_taco_data(location_string: str) -> pd.DataFrame:
    """
    Queries Google Places Text Search API to pull local taco establishments.
    Flattens the nested JSON response into a clean, flat Pandas DataFrame.
    """
    if not API_KEY:
        raise ValueError("Missing GOOGLE_PLACES_API_KEY. Please verify your .env file.")

    # Modern Google Places Text Search Endpoint
    url = "https://places.googleapis.com/v1/places:searchText"
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        # FieldMasking tells Google EXACTLY what keys to return. This saves memory and budget.
        "X-Goog-FieldMask": "places.displayName,places.rating,places.userRatingCount,places.priceLevel,places.location,places.formattedAddress"
    }
    
    json_data = {
        "textQuery": f"tacos in {location_string}",
        "languageCode": "en",
        "maxResultCount": 20 # Pulls top 20 relevant results
    }
    
    try:
        response = requests.post(url, headers=headers, json=json_data)
        if response.status_code != 200:
            print(f"Google API Error Status {response.status_code}: {response.text}")
            return pd.DataFrame()
            
        places_data = response.json().get('places', [])
        
        if not places_data:
            print(f"No taco spots returned for location: {location_string}")
            return pd.DataFrame()
            
        # Parse through the complex JSON arrays to create a flat dictionary list
        flattened_records = []
        for place in places_data:
            price_str = place.get('priceLevel', 'PRICE_LEVEL_MODERATE')
            
            record = {
                "name": place.get('displayName', {}).get('text', 'Unknown Taco Shop'),
                "rating": place.get('rating', np.nan if 'np' in globals() else 3.5),
                "review_count": place.get('userRatingCount', 0),
                "price_level_raw": price_str,
                "lat": place.get('location', {}).get('latitude'),
                "lon": place.get('location', {}).get('longitude'),
                "address": place.get('formattedAddress', '')
            }
            flattened_records.append(record)
            
        return pd.DataFrame(flattened_records)

    except Exception as e:
        print(f"Connection Exception Encountered: {e}")
        return pd.DataFrame()

# ---- Local Diagnostic Test Run ----
if __name__ == "__main__":
    # Import numpy natively just for the fallback default values in local testing
    import numpy as np
    
    print("Executing local test pipeline for 'La Jolla, San Diego'...")
    test_df = fetch_local_taco_data("La Jolla, San Diego")
    
    if not test_df.empty:
        print("\nSuccess! Data pipeline running perfectly. Sample Output:")
        print(test_df[['name', 'rating', 'review_count', 'price_level_raw']].head())
    else:
        print("\n Pipeline returned an empty dataframe. Double check your API key.")