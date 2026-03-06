import pandas as pd

ratings = pd.read_csv(
    'ml-1m/ratings.dat', 
    sep='::', 
    engine='python', 
    names=['userId', 'itemId', 'rating', 'timestamp'],
    encoding='ISO-8859-1'
)

movies = pd.read_csv(
    'ml-1m/movies.dat', 
    sep='::', 
    engine='python', 
    names=['itemId', 'title', 'genres'],
    encoding='ISO-8859-1'
)

users = pd.read_csv(
    'ml-1m/users.dat', 
    sep='::', 
    engine='python', 
    names=['userId', 'gender', 'age', 'occupation', 'zip-code'],
    encoding='ISO-8859-1'
)

ratings['userId'] = ratings['userId'].astype(int)
ratings['itemId'] = ratings['itemId'].astype(int)
ratings['rating'] = ratings['rating'].astype(float)
ratings.to_parquet('data/interactions.parquet', index=False)
movies.to_parquet('data/items.parquet', index=False)

print("OK")