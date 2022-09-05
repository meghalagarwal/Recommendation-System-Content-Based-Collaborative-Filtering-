'''
Collaborative filters can further be classified into two types:

## User-based Filtering:
These systems recommend products to a user that similar users have liked.
For example, let's say Alice and Bob have a similar interest in books (that is, they largely like and dislike the same books).
Now, let's say a new book has been launched into the market, and Alice has read and loved it.
It is, therefore, highly likely that Bob will like it too, and therefore, the system recommends this book to Bob.

## Item-based Filtering:
These systems are extremely similar to the content recommendation engine that you built.
These systems identify similar items based on how people have rated it in the past.
For example, if Alice, Bob, and Eve have given 5 stars to The Lord of the Rings and The Hobbit, the system identifies the items as similar.
Therefore, if someone buys The Lord of the Rings, the system also recommends The Hobbit to him or her.
'''

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

# Load Metadata
header1 = ['user_id','item_id','rating','timestamp']
dataset = pd.read_csv('/home/meghal/Personal/Projects/Recommendation System/Data/ml-100k/u.data',sep = '\t',names = header1 )
print(dataset.head())

header2 = ['id', 'Movie Name', 'Release Date', 'link', 'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16','A17','A18','A19', 'A20']
movies = pd.read_csv('/home/meghal/Personal/Projects/Recommendation System/Data/ml-100k/u.item', sep = '|', names = header2, encoding='latin-1' )
movies.drop('id', axis='columns')
print(movies[['Movie Name']])

# Transforming data into the matrix
n_users = dataset.user_id.unique().shape[0]
n_items = dataset.item_id.unique().shape[0]
n_items = dataset['item_id'].max()

A = np.zeros((n_users,n_items))

for line in dataset.itertuples():
    A[line[1]-1,line[2]-1] = line[3]

print("Original rating matrix : ",A)

for i in range(len(A)):
  for j in range(len(A[0])):
    if A[i][j]>=3:
      A[i][j]=1
    else:
      A[i][j]=0

csr_sample = csr_matrix(A)
print(csr_sample)

# Items Similarity Computation
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=3, n_jobs=-1)
knn.fit(csr_sample)

# Generate Recommendations
dataset_sort_des = dataset.sort_values(['user_id', 'timestamp'], ascending=[True, False])
filter1 = dataset_sort_des[dataset_sort_des['user_id'] == 1].item_id
filter1 = filter1.tolist()
filter1 = filter1[:20]

print("Items liked by user: ",filter1)

distances1=[]
indices1=[]

for i in filter1:
  distances , indices = knn.kneighbors(csr_sample[i],n_neighbors=3)
  indices = indices.flatten()
  indices= indices[1:]
  indices1.extend(indices)

print("Items to be recommended: ",indices1)

'''
The output screen shows the recommendations being generated for user1.
For ease of use and simplicity, we have used movie_id here but movie_id can be replaced with corresponding movie name by fetching information from the movies dataset.
'''

for ids in indices1:
    print("Recommended Movies: ", movies.iloc[ids-1]['Movie Name'])