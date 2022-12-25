from Low_rank_Matrix_factorization import MF
import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('Recommendation System/ml-1m/ratings.dat', sep='::', names=r_cols, encoding='latin-1')
ratings = ratings_base.to_numpy()

ratings[:, :2] -= 1 # indices start from 0

from sklearn.model_selection import train_test_split

rate_train, rate_test = train_test_split(ratings, test_size=0.33, random_state=42)
#print(rate_train.shape, rate_test.shape)

rs = MF(rate_train, K = 2, lam = 0.1, print_every = 2, learning_rate = 2, max_iter = 10, user_based = 0)
rs.fit()
RMSE = rs.evaluate_RMSE(rate_test) 
print('Item-based MF, RMSE =', RMSE)