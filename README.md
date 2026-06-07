# 🎬 Movie Recommendation System

A content-based movie recommendation system built using Machine Learning and Streamlit.

## Features

* Recommend similar movies based on content
* TF-IDF Vectorization
* Cosine Similarity
* TMDB API integration for movie posters
* Interactive Streamlit user interface
* Dark-themed UI design

## Technologies Used

* Python
* Streamlit
* Pandas
* Scikit-Learn
* TMDB API
* Natural Language Processing (NLP)

## How It Works

The system combines movie overviews, genres, and keywords into a feature set. TF-IDF Vectorization converts text into numerical vectors, and Cosine Similarity is used to identify movies with similar content.

## Installation

1. Clone the repository

```bash
git clone https://github.com/Inchara-hub16/Movie-recommendation-system.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Create a `.env` file

```env
TMDB_API_KEY=YOUR_API_KEY
```

4. Run the application

```bash
streamlit run movie.py
```

## Future Improvements

* User-based recommendations
* Hybrid recommendation system
* Movie ratings integration
* Personalized recommendations

## Author

Inchara Manojkumar
