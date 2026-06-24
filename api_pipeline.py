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
    
    # Updated FieldMask to explicitly request dietary booleans
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": (
            "places.displayName,places.rating,places.userRatingCount,places.priceLevel,"
            "places.vegetarianServed,places.veganServed,places.location,places.formattedAddress"
        )
    }
    
    json_data = {
        "textQuery": f"best vegetarian restaurants in {location_string}",
        "languageCode": "en",
        "maxResultCount": 20 
    }
    
    try:
        response = requests.post(url, headers=headers, json=json_data, timeout=5.0)
        if response.status_code != 200:
            print(f"Google API Error Status {response.status_code}: {response.text}")
            return pd.DataFrame()
            
        places_data = response.json().get('places', [])
        
        if not places_data:
            print(f"No vegetarian spots returned for location: {location_string}")
            return pd.DataFrame()
            
        flattened_records = []
        for place in places_data:
            price_str = place.get('priceLevel', 'PRICE_LEVEL_MODERATE')
            
            record = {
                "name": place.get('displayName', {}).get('text', 'Unknown Restaurant'),
                "rating": place.get('rating', np.nan),
                "review_count": place.get('userRatingCount', 0),
                "price_level_raw": price_str,
                "vegetarian_served": place.get('vegetarianServed', False),
                "vegan_served": place.get('veganServed', False),
                "lat": place.get('location', {}).get('latitude'),
                "lon": place.get('location', {}).get('longitude'),
                "address": place.get('formattedAddress', '')
            }
            flattened_records.append(record)
            
        return pd.DataFrame(flattened_records)

    except requests.exceptions.Timeout:
        print("Timeout Error: Google's server took too long to respond.")
        return pd.DataFrame()
        
    except requests.exceptions.ConnectionError:
        print("Connection Error: Verified complete network interface drop.")
        return pd.DataFrame()
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP Protocol Error: Server responded with bad status: {http_err}")
        return pd.DataFrame()

if __name__ == "__main__":
    print("Executing local test pipeline for 'La Jolla, San Diego'...")
    test_df = fetch_local_vegetarian_data("La Jolla, San Diego")
    
    if not test_df.empty:
        print("\nSuccess! Vegetarian data pipeline running perfectly. Sample Output:")
        print(test_df[['name', 'rating', 'review_count', 'vegetarian_served', 'vegan_served']].head())
    else:
        print("\nPipeline returned an empty dataframe. Double check your API key and network connection.")