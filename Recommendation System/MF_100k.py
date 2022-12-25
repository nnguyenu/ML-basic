from Low_rank_Matrix_factorization import MF
import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('Recommendation System/ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('Recommendation System/ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.to_numpy()
rate_test = ratings_test.to_numpy()

# indices start from 0
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

def user_based_MF():
    rs = MF(rate_train, K = 10, lam = .1, print_every = 10, learning_rate = 0.75, max_iter = 100, user_based = 1)
    rs.fit()
    RMSE = rs.evaluate_RMSE(rate_test)
    print('User-based MF, RMSE =', RMSE)

def item_based_MF():
    rs = MF(rate_train, K = 10, lam = .1, print_every = 10, learning_rate = 0.75, max_iter = 100, user_based = 0)
    rs.fit()
    RMSE = rs.evaluate_RMSE(rate_test)
    print('Item-based MF, RMSE =', RMSE)

def no_regularization():
    rs = MF(rate_train, K = 2, lam = 0, print_every = 10, learning_rate = 1, max_iter = 100, user_based = 0)
    rs.fit()
    RMSE = rs.evaluate_RMSE(rate_test)
    print('Item-based MF, RMSE =', RMSE)

no_regularization()
