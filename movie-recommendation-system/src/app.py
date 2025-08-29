import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_data
from recommendation import generate_movie_embeddings, recommend_movies

# --- Page Configuration ---
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

DATASET_PATH = os.path.join('..', 'imdb_top_1000.csv')

# --- State Management ---
if 'data' not in st.session_state:
    st.session_state.data = load_data(DATASET_PATH)

if 'embeddings' not in st.session_state and st.session_state.data is not None:
    with st.spinner('Warming up the recommendation engine... This may take a moment.'):
        st.session_state.embeddings = generate_movie_embeddings(st.session_state.data.copy())

# --- UI Layout ---
st.title("🎬 Transformers-Powered Movie Recommender")
st.markdown(
    "Welcome! Describe the kind of movie you're in the mood for. "
    'Mention movies you like in double quotes (e.g., `"The Dark Knight"`) '
    'and set a minimum rating (e.g., `imdb:8.5`).'
)

# --- Main Application ---
if st.session_state.data is None:
    st.error(f"Failed to load the dataset. Please make sure `imdb_top_1000.csv` is in the '{PROJECT_DIR}' directory.")
else:
    with st.form(key='movie_input_form'):
        user_prompt = st.text_input(
            "Enter your movie preferences:",
            'I want a thought-provoking sci-fi movie like "Inception" with a rating of at least imdb:8'
        )
        submit_button = st.form_submit_button(label='Get Recommendations')

    if submit_button and user_prompt:
        with st.spinner('Finding the perfect movies for you...'):
            recommendations = recommend_movies(user_prompt, st.session_state.data, st.session_state.embeddings)

        st.subheader("Here are your personalized recommendations:")

        if not recommendations.empty:
            for index, row in recommendations.iterrows():
                st.markdown(f"### {row['Title']} ({row['IMDB_Rating']} ⭐)")
                st.markdown(f"**Genre:** {row['Genre']}")
                st.markdown(f"> {row['Overview']}")
                st.markdown("---")
        else:
            st.warning("Couldn't find any movies matching your criteria. Please try a different query.")