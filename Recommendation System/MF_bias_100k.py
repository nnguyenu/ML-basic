from LR_MF_with_bias import MF_bias
import pandas as pd


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('Recommendation System/ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('Recommendation System/ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.to_numpy()
rate_test = ratings_test.to_numpy()

# indices start from 0
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

rs = MF_bias(rate_train, K = 100, lam = .01, print_every = 20, learning_rate = 2, max_iter = 200)
rs.fit()
RMSE = rs.evaluate_RMSE(rate_test)
print('User-based MF, RMSE =', RMSE)