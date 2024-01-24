import pandas as pd
import streamlit as st

DATA_PATH = "data/processed_data.csv"
COLS = ["book_title", "author", "genre", "publication_year", "summary"]

st.set_page_config(
    page_title="Books Collection",
    page_icon=":open_book:",
    layout="wide"
)

#st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, sep=";")
    data = data[COLS]
    return data

st.title("Our Library")
data = load_data()

st.markdown(f"""
    **Welcome to our literary haven!**
            
    \nDive into our huge collection of books, where you will find {data.shape[0]} titles from {data.genre.nunique()} different genres. 
    Whether you're on a quest for thrilling adventures, thought-provoking narratives, or timeless classics, we have a treasure trove for every book enthusiast.
    Explore, discover, and lose yourself in the world of words.
""")


titles = data["book_title"].to_list()
genres = data["genre"].apply(eval).explode().drop_duplicates().sort_values().to_list()
authors = data["author"].value_counts().index.drop_duplicates().to_list()
years = data["publication_year"].dropna().astype(int).to_list()
max_year, min_year = max(years), min(years)

title_search = st.sidebar.selectbox(
   "Title",
   titles,
   index=None,
   placeholder="Select book title...",
)

selected_authors = st.sidebar.multiselect(
    "Authors",
    authors,
    max_selections=5,
    placeholder="Select authors...",
)

selected_genres = st.sidebar.multiselect(
    "Genre",
    genres,
    max_selections=6,
    placeholder="Select genres...",
)

selected_years = st.sidebar.slider(
    'Select year of publication',
    min_year, max_year, (min_year, max_year))

st.markdown("### Selected titles")

filtered_data = data.copy()

if title_search is not None:
    filtered_data = filtered_data[filtered_data["book_title"] == title_search]

if len(selected_authors) > 0:
    filtered_data = filtered_data[filtered_data["author"].isin(selected_authors)]

if len(selected_genres) > 0:
    filtered_data = filtered_data[filtered_data['genre'].apply(lambda x: any(item in x for item in selected_genres))]

if selected_years != (min_year, max_year):
    filtered_data = filtered_data[filtered_data["publication_year"].between(*selected_years)]

filtered_data["genre"] = filtered_data.genre.apply(eval).str.join(", ").replace("", "-")
filtered_data["author"] = filtered_data.author.fillna("Unkown")
filtered_data = filtered_data.style.format(
    {
        "publication_year": lambda x: "{:,.1f}".format(x)
    },
    thousands=""
)

st.dataframe(filtered_data, hide_index=True)

# for index, row in filtered_data.iterrows():
#     st.write(f"**Title:** {row['book_title']}")
#     st.write(f"**Author:** {row['author']}")
#     st.write(f"**Genre:** {', '.join(row['genre']) if row['genre'] else 'N/A'}")
#     st.write(f"**Publication Year:** {row['publication_year'] if pd.notnull(row['publication_year']) else 'N/A'}")
#     #st.write(f"**Summary:** {row['summary']}")
#     st.markdown('---')
    