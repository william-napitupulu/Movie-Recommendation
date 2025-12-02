import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
import os

def download_and_load_data():
    """Download and load MovieLens dataset"""
    # Check if file exists, if not download
    if not os.path.exists('ml-latest-small.zip'):
        print("Downloading dataset...")
        import urllib.request
        url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
        urllib.request.urlretrieve(url, 'ml-latest-small.zip')
        print("Download complete.")
    
    if os.path.exists('ml-latest-small.zip'):
        with zipfile.ZipFile('ml-latest-small.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
    
    base_dir = 'ml-latest-small'
    movies = pd.read_csv(f'{base_dir}/movies.csv')
    ratings = pd.read_csv(f'{base_dir}/ratings.csv')
    links = pd.read_csv(f'{base_dir}/links.csv')
    tags = pd.read_csv(f'{base_dir}/tags.csv')
    
    return movies, ratings, links, tags

def prepare_content_based_data(movies, ratings):
    """Prepare data for content-based filtering"""
    print("Preparing data for Content-Based Filtering...")
    
    # Merge ratings with movie info
    all_movie = pd.merge(ratings, movies[['movieId', 'title', 'genres']], on='movieId')
    
    # Remove missing values
    all_movie_clean = all_movie.dropna()
    
    # Create sorted dataframe
    fix_movie = all_movie_clean.sort_values('movieId', ascending=True)
    
    # Create deduplicated catalog for content-based filtering
    data = fix_movie.drop_duplicates('movieId')
    data = data.sort_values('movieId')
    data = data.reset_index(drop=True)
    
    # Clean genre text (replace | with space)
    data['genres'] = data['genres'].str.replace('|', ' ')
    
    print(f"Prepared {len(data)} unique movies for content-based filtering")
    return data

def content_based_filtering(data):
    """Build content-based filtering model using TF-IDF and Cosine Similarity"""
    print("Building Content-Based Filtering model...")
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['genres'])
    
    # Compute Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix)
    cosine_sim_df = pd.DataFrame(
        cosine_sim, 
        index=data['title'], 
        columns=data['title']
    )
    
    print(f"Cosine similarity matrix shape: {cosine_sim_df.shape}")
    return cosine_sim_df, data

def get_content_recommendations(nama_film, similarity_data, items, k=5):
    """Get top-k movie recommendations based on content similarity"""
    try:
        index = similarity_data.loc[:, nama_film].to_numpy().argpartition(
            range(-1, -k, -1)
        )
        closest = similarity_data.columns[index[-1:-(k+2):-1]]
        closest = closest.drop(nama_film, errors='ignore')
        return pd.DataFrame(closest).merge(items[['title', 'genres']]).head(k)
    except KeyError:
        return f"Film '{nama_film}' tidak ditemukan dalam dataset."

class RecommenderNet(tf.keras.Model):
    """Neural Network model for Collaborative Filtering with Embeddings"""
    
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        
        # User Embedding Layer
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.user_bias = layers.Embedding(num_users, 1)
        
        # Movie Embedding Layer
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=keras.regularizers.l2(1e-6)
        )
        self.movie_bias = layers.Embedding(num_movies, 1)
    
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        
        x = dot_user_movie + user_bias + movie_bias
        
        return tf.nn.sigmoid(x)

def prepare_collaborative_data(ratings):
    """Prepare data for collaborative filtering"""
    print("Preparing data for Collaborative Filtering...")
    
    # Encode user IDs
    user_ids = ratings['userId'].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}
    
    # Encode movie IDs
    movie_ids = ratings['movieId'].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movieencoded2movie = {i: x for i, x in enumerate(movie_ids)}
    
    # Map to dataframe
    ratings['user'] = ratings['userId'].map(user2user_encoded)
    ratings['movie'] = ratings['movieId'].map(movie2movie_encoded)
    
    # Get counts
    num_users = len(user2user_encoded)
    num_movies = len(movie2movie_encoded)
    
    # Normalize ratings
    ratings['rating'] = ratings['rating'].values.astype(np.float32)
    min_rating = min(ratings['rating'])
    max_rating = max(ratings['rating'])
    
    print(f"Number of users: {num_users}, Number of movies: {num_movies}")
    print(f"Min rating: {min_rating}, Max rating: {max_rating}")
    
    # Shuffle data
    df = ratings.sample(frac=1, random_state=42)
    
    # Prepare features and target
    x = df[['user', 'movie']].values
    y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    
    # Train-test split
    train_indices = int(0.8 * df.shape[0])
    x_train, x_val = x[:train_indices], x[train_indices:]
    y_train, y_val = y[:train_indices], y[train_indices:]
    
    return (num_users, num_movies, x_train, x_val, y_train, y_val,
            user2user_encoded, userencoded2user, 
            movie2movie_encoded, movieencoded2movie)

def collaborative_filtering(num_users, num_movies, x_train, x_val, y_train, y_val, 
                           epochs=20, batch_size=8):
    """Build and train collaborative filtering model"""
    print("Building Collaborative Filtering model...")
    
    # Initialize model
    model = RecommenderNet(num_users, num_movies, embedding_size=50)
    
    # Compile model
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )
    
    # Train model
    print(f"Training model for {epochs} epochs...")
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=1
    )
    
    return model, history

def get_collaborative_recommendations(user_id, model, ratings, movies,
                                     user2user_encoded, movie2movie_encoded,
                                     movieencoded2movie, top_n=10):
    """Get top-n movie recommendations for a user using collaborative filtering"""
    
    # Get movies watched by user
    movie_watched_by_user = ratings[ratings.userId == user_id]
    
    # Get movies not watched
    movie_not_watched = movies[~movies['movieId'].isin(movie_watched_by_user.movieId.values)]['movieId']
    movie_not_watched = list(
        set(movie_not_watched).intersection(set(movie2movie_encoded.keys()))
    )
    
    # Encode movies
    movie_not_watched = [[movie2movie_encoded.get(x)] for x in movie_not_watched]
    user_encoder = user2user_encoded.get(user_id)
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movie_not_watched), movie_not_watched)
    )
    
    # Predict ratings
    ratings_pred = model.predict(user_movie_array).flatten()
    
    # Get top-n indices
    top_ratings_indices = ratings_pred.argsort()[-top_n:][::-1]
    recommended_movie_ids = [
        movieencoded2movie.get(movie_not_watched[x][0]) for x in top_ratings_indices
    ]
    
    # Get movie info
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]
    
    return recommended_movies[['title', 'genres']]

if __name__ == "__main__":
    try:
        print("="*60)
        print("MOVIE RECOMMENDATION SYSTEM")
        print("="*60)
        
        # Load data
        movies, ratings, links, tags = download_and_load_data()
        
        # ==================== CONTENT-BASED FILTERING ====================
        print("\n" + "="*60)
        print("CONTENT-BASED FILTERING")
        print("="*60)
        
        # Prepare and build content-based model
        data = prepare_content_based_data(movies, ratings)
        cosine_sim_df, data = content_based_filtering(data)
        
        # Get recommendations
        test_movie = 'Toy Story (1995)'
        print(f"\nTop-5 recommendations for '{test_movie}':")
        print("-"*60)
        recommendations = get_content_recommendations(
            test_movie, 
            cosine_sim_df, 
            data,
            k=5
        )
        print(recommendations.to_string(index=False))
        
        # ==================== COLLABORATIVE FILTERING ====================
        print("\n" + "="*60)
        print("COLLABORATIVE FILTERING")
        print("="*60)
        
        # Prepare data
        (num_users, num_movies, x_train, x_val, y_train, y_val,
         user2user_encoded, userencoded2user, 
         movie2movie_encoded, movieencoded2movie) = prepare_collaborative_data(ratings)
        
        # Train model (reduced epochs for demo)
        model, history = collaborative_filtering(
            num_users, num_movies, 
            x_train, x_val, y_train, y_val,
            epochs=20,  # Matches notebook configuration
            batch_size=8
        )
        
        # Get recommendations for a sample user
        sample_user_id = ratings.userId.sample(1).iloc[0]
        print(f"\nTop-10 recommendations for User {sample_user_id}:")
        print("-"*60)
        
        collab_recommendations = get_collaborative_recommendations(
            sample_user_id,
            model,
            ratings,
            movies,
            user2user_encoded,
            movie2movie_encoded,
            movieencoded2movie,
            top_n=10
        )
        print(collab_recommendations.to_string(index=False))
        
        # Display final metrics
        print("\n" + "="*60)
        print("TRAINING RESULTS")
        print("="*60)
        final_train_rmse = history.history['root_mean_squared_error'][-1]
        final_val_rmse = history.history['val_root_mean_squared_error'][-1]
        print(f"Final Training RMSE: {final_train_rmse:.4f}")
        print(f"Final Validation RMSE: {final_val_rmse:.4f}")
        print("="*60)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
