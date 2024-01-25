import os
import sys
sys.path.insert(1, os.path.abspath("./"))
from typing import Union

import numpy as np
import pandas as pd
import streamlit as st

from scripts.merge_rankings import merge_rankings
from scripts.collaborative_model import CollaborativeModel


DATA_PATH = "data/processed_data.csv"
REVIEWS_DATA_PATH =  "data/reviews.csv"
COLS = ["book_title", "author", "genre", "publication_year", "summary"]

st.set_page_config(
    page_title="Books Reccomender",
    page_icon=":open_book:",
    #layout="wide"
)

def get_recommendations(sim_matrix: np.ndarray, idx: Union[int, list[int]]) -> list[int]:
        if isinstance(idx, int):
            means = sim_matrix[idx]
        else:
            means = np.mean(sim_matrix[idx], axis=0)
        sim_scores = list(enumerate(means))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        idx = [idx] if isinstance(idx, int) else idx
        indices = [i[0] for i in sim_scores if i[0] not in idx]
        return indices[:10]

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, sep=";")
    data = data[COLS]
    return data

@st.cache_data
def get_sim_matrix(sim_matrix_path: str):
    sim_matrix = np.load(sim_matrix_path)
    sim_matrix = sim_matrix["arr_0"]
    return sim_matrix

@st.cache_data
def get_reviews_data():
    return pd.read_csv(REVIEWS_DATA_PATH, sep=";", index_col=0)


st.title("Books reccomedner")
data = load_data()
reviews_data = get_reviews_data()
sim_matrix = get_sim_matrix("data/sim_matrix/doc2vec_sim_matrix.npy")
collaboartive_model = CollaborativeModel(reviews_data)

titles = data["book_title"].to_list()

st.markdown("### 1. Select your preferences")

col1, col2, col3 = st.columns([1, 7, 1])

with col1:
    st.write("Opinions")

with col2:
    preference = st.slider(
    '<-->',
    0, 100, 50, label_visibility='hidden')

with col3:
    st.write("Content")


st.markdown("### 2. Select titles")

selected_books = st.multiselect(
    "Books",
    titles,
    max_selections=5,
    placeholder="Select books you like...",
    label_visibility='hidden'
)

books_ids = data.query("book_title.isin(@selected_books)").index

show_selected = st.checkbox('Show selected books')

if show_selected:
    st.write("Selected books")

    selected_books = data.iloc[books_ids, :]
    
    # stylistic changes for displaing data
    selected_books["genre"] = selected_books.genre.apply(eval).str.join(", ").replace("", "-")
    selected_books["author"] = selected_books.author.fillna("Unkown")
    selected_books = selected_books.style.format(
        {
            "publication_year": lambda x: "{:,.1f}".format(x)
        },
        thousands=""
    )

    st.dataframe(selected_books, hide_index=True)

if st.button("Show reccomendations", type="primary"):
    # recoms based on content
    content_recommended_ids = get_recommendations(sim_matrix, books_ids)

    # recoms based on opinions
    my_scores = {k: 5 for k in books_ids}
    reviews_recommended_ids = collaboartive_model.get_recommendations(my_scores)

    # merged recoms
    recommended_ids = merge_rankings(reviews_recommended_ids, content_recommended_ids, preference)

    recommended_books = data.iloc[recommended_ids, :]
    st.write("Recommended books")
    
    # stylistic changes for displaing data
    recommended_books["genre"] = recommended_books.genre.apply(eval).str.join(", ").replace("", "-")
    recommended_books["author"] = recommended_books.author.fillna("Unkown")
    filtered_data = recommended_books.style.format(
        {
            "publication_year": lambda x: "{:,.1f}".format(x)
        },
        thousands=""
    )
    st.dataframe(recommended_books, hide_index=True)