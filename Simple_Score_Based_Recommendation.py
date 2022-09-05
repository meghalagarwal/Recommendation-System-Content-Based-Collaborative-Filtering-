'''
## Simple Recommenders
As described in the previous section, simple recommenders are basic systems that recommend the top items based on a certain metric or score.
In this section, we will build a simplified clone of IMDB Top 250 Movies using metadata collected from IMDB.

The following are the steps involved:
Decide on the metric or score to rate movies on.
Calculate the score for every movie.
Sort the movies based on the score and output the top results.
'''
# Import Pandas
import pandas as pd

# Load Movies Metadata
metadata = pd.read_csv('/home/meghal/Personal/Projects/Recommendation System/Data/movies_metadata.csv', low_memory=False)

# Print the first three rows
metadata.head(3)

# Calculate mean of vote average column
C = metadata['vote_average'].mean()
print(C)

# Calculate the minimum number of votes required to be in the chart, m
m = metadata['vote_count'].quantile(0.90)
print(m)

# Filter out all qualified movies into a new DataFrame
q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
q_movies.shape

# Function that computes the weighted rating of each movie
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(20)