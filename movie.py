import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import ast
import requests
import os

from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM DARK BLUE THEME
# ---------------------------------------------------

st.markdown(
    """
    <style>

    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: white;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }

    /* Paragraph Text */
    p {
        color: white;
    }

    /* Selectbox */
    div[data-baseweb="select"] {
        background-color: #1F2937;
        border-radius: 10px;
        color: white;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2563EB;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
        transition: 0.3s;
    }

    /* Button Hover */
    .stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
    }

    /* Sidebar Text */
    section[data-testid="stSidebar"] * {
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.markdown(
    """
    <h1 style='text-align: center;'>
        🎬 Movie Recommendation System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

movies = pd.read_csv("tmdb_5000_movies.csv")

# Select useful columns
movies = movies[['title', 'overview', 'genres', 'keywords']]

# Remove missing values
movies.dropna(inplace=True)

# ---------------------------------------------------
# CONVERT FUNCTION
# ---------------------------------------------------

def convert(text):

    L = []

    for i in ast.literal_eval(text):
        L.append(i['name'])

    return L

# Apply conversion
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Convert lists into strings
movies['genres'] = movies['genres'].apply(lambda x: " ".join(x))
movies['keywords'] = movies['keywords'].apply(lambda x: " ".join(x))

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------

movies['tags'] = (
    movies['overview'] + " " +
    movies['genres'] + " " +
    movies['keywords']
)

# Convert tags to lowercase
movies['tags'] = movies['tags'].apply(lambda x: x.lower())

# ---------------------------------------------------
# TF-IDF VECTORIZATION
# ---------------------------------------------------

tfidf = TfidfVectorizer(stop_words='english')

vectors = tfidf.fit_transform(movies['tags'])

# ---------------------------------------------------
# COSINE SIMILARITY
# ---------------------------------------------------

similarity = cosine_similarity(vectors)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("📌 About Project")

st.sidebar.write(
    """
    This is a Content-Based Movie Recommendation System
    built using:

    ✅ Python  
    ✅ Streamlit  
    ✅ TF-IDF Vectorizer  
    ✅ Cosine Similarity  
    ✅ NLP Techniques  
    ✅ TMDB API Integration  
    """
)

st.sidebar.markdown("---")

st.sidebar.info(f"🎥 Total Movies: {movies.shape[0]}")

# ---------------------------------------------------
# RECOMMENDATION FUNCTION
# ---------------------------------------------------

def recommend(movie):

    movie_list = movies['title'].str.lower()

    if movie.lower() not in movie_list.values:
        st.error("Movie not found!")
        return []

    # Movie index
    movie_index = movies[
        movies['title'].str.lower() == movie.lower()
    ].index[0]

    # Similarity scores
    distances = similarity[movie_index]

    # Sort movies
    recommended_movies = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return recommended_movies

# ---------------------------------------------------
# FETCH MOVIE POSTER
# ---------------------------------------------------

def fetch_poster(movie_title):

    url = (
        f"https://api.themoviedb.org/3/search/movie"
        f"?api_key={API_KEY}&query={movie_title}"
    )

    data = requests.get(url).json()

    if (
        data['results']
        and data['results'][0]['poster_path']
    ):

        poster_path = data['results'][0]['poster_path']

        full_path = (
            "https://image.tmdb.org/t/p/w500/"
            + poster_path
        )

        return full_path

    return None

# ---------------------------------------------------
# MOVIE DROPDOWN
# ---------------------------------------------------

movie_options = ["Select a movie"] + list(movies['title'].values)

selected_movie = st.selectbox(
    "🎥 Select a Movie",
    movie_options
)

# ---------------------------------------------------
# BUTTON
# ---------------------------------------------------

if st.button("Recommend Movies"):

    if selected_movie == "Select a movie":

        st.warning("Please select a movie first.")

    else:

        with st.spinner("Finding best recommendations..."):

            recommendations = recommend(selected_movie)

            if recommendations:

                st.markdown("---")

                st.subheader(
                    f"Top 5 Movies Similar to {selected_movie}"
                )

                # Create columns
                cols = st.columns(5)

                # ---------------------------------------------------
                # DISPLAY MOVIE CARDS
                # ---------------------------------------------------

                for idx, i in enumerate(recommendations):

                    with cols[idx]:

                        # Movie title
                        movie_title = movies.iloc[i[0]].title

                        # Genres
                        movie_genres = movies.iloc[i[0]].genres

                        # Overview preview
                        movie_overview = (
                            movies.iloc[i[0]].overview[:120] + "..."
                        )

                        # Format genres
                        genre_text = movie_genres.replace(" ", " | ")

                        # Fetch poster
                        poster_url = fetch_poster(movie_title)

                        # Show poster
                        if poster_url:
                            st.image(poster_url)

                        # Movie card HTML
                        card_html = f"""
                        <div style="
                            background: linear-gradient(
                                135deg,
                                #1E3A8A,
                                #111827
                            );
                            padding:15px;
                            border-radius:18px;
                            height:250px;
                            box-shadow: 0px 6px 18px rgba(0,0,0,0.4);
                            overflow:hidden;
                            color:white;
                            font-family:Arial;
                        ">

                            <h3 style="
                                text-align:center;
                                margin-bottom:10px;
                            ">
                                {idx+1}. {movie_title}
                            </h3>

                            <p style="
                                color:#93C5FD;
                                font-size:13px;
                                text-align:center;
                                font-weight:bold;
                            ">
                                {genre_text}
                            </p>

                            <hr style="border:1px solid #374151;">

                            <p style="
                                font-size:14px;
                                text-align:justify;
                                line-height:1.5;
                            ">
                                {movie_overview}
                            </p>

                        </div>
                        """

                        components.html(card_html, height=260)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")