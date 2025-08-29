import torch
import numpy as np
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from utils import preprocess_text

# Load pre-trained model tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    """Generates a BERT embedding for the given text."""
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[0]
        embedding = hidden_states[:, 0, :].numpy()
    return embedding

def generate_movie_embeddings(df):
    """Generates and returns BERT embeddings for all movie overviews."""
    df['Combined_Text'] = df['Title'] + ". " + df['Overview']
    df['Processed_Text'] = df['Combined_Text'].apply(preprocess_text)

    embeddings = df['Processed_Text'].apply(get_bert_embedding)
    return np.vstack(embeddings)

def recommend_movies(prompt, df, movie_embeddings):
    """Recommends movies based on a user prompt."""
    from utils import extract_user_preferences

    liked_movies, min_rating = extract_user_preferences(prompt)

    processed_prompt = preprocess_text(prompt)
    prompt_embedding = get_bert_embedding(processed_prompt)

    similarity_scores = cosine_similarity(prompt_embedding, movie_embeddings).flatten()

    if liked_movies:
        for movie_title in liked_movies:
            if movie_title in df['Title'].values:
                movie_index = df.index[df['Title'] == movie_title][0]
                similarity_scores[movie_index] *= 1.5

    df_recommendations = df.copy()
    df_recommendations['similarity'] = similarity_scores

    if min_rating > 0:
        df_recommendations = df_recommendations[df_recommendations['IMDB_Rating'] >= min_rating]

    df_recommendations = df_recommendations.sort_values(by='similarity', ascending=False)
    df_recommendations = df_recommendations[~df_recommendations['Title'].isin(liked_movies)]

    return df_recommendations.head(15)