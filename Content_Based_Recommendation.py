# Import Pandas
import pandas as pd
# Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Load Movies Metadata
metadata = pd.read_csv('/home/meghal/Personal/Projects/Recommendation System/Data/movies_metadata.csv', low_memory=False)

#Print plot overviews of the first 5 movies.
metadata['overview'].head()

'''
You will compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each document.
This will give you a matrix where each column represents a word in the overview vocabulary (all the words that appear in at least one document),
and each column represents a movie.
'''

# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

# Output the shape of tfidf_matrix
tfidf_matrix.shape

# Array mapping from feature integer indices to feature name.
tfidf.get_feature_names_out()[5000:5010]

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim.shape

cosine_sim[1]

# Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return metadata['title'].iloc[movie_indices]

get_recommendations('The Dark Knight Rises')

get_recommendations('The Godfather')