import pandas as pd
import re

def load_data(file_path):
    """
    Loads the IMDb movie dataset from a CSV file and performs initial cleaning.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing the cleaned movie data.
    """
    try:
        df = pd.read_csv(file_path)
        # Keep only the essential columns for the recommendation logic
        df = df[['Series_Title', 'Genre', 'IMDB_Rating', 'Overview']].copy()
        df.rename(columns={'Series_Title': 'Title'}, inplace=True)
        return df
    except FileNotFoundError:
        return None

def preprocess_text(text):
    """
    Cleans and normalizes a given text string.

    Args:
        text (str): The input string to clean.

    Returns:
        str: The cleaned and lowercased string.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove special characters
    return text

def extract_user_preferences(prompt):
    """
    Parses the user's input prompt to extract movie titles and a minimum IMDb rating.

    Args:
        prompt (str): The user's input string.

    Returns:
        tuple: A tuple containing a list of movie titles and the IMDb rating (float).
    """
    movies = re.findall(r'"(.*?)"', prompt)
    rating_match = re.search(r'imdb:(\d+\.?\d*)', prompt)
    imdb_rating = float(rating_match.group(1)) if rating_match else 0.0
    return movies, imdb_rating