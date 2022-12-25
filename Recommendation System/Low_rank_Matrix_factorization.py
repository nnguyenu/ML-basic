import pandas as pd 
import numpy as np

class MF(object):
    def __init__(self, Y_data, K, lam = 0.1, Xinit = None, Winit = None, learning_rate = 0.5, 
                max_iter = 1000, print_every = 100, user_based = 1):
        self.Y_raw_data = Y_data
        self.K = K
        self.lam = lam      # regularization parameter
        self.learning_rate = learning_rate  # learning rate for gradient descent
        self.max_iter = max_iter    # maximum number of iterations
        self.print_every = print_every  # print results after 'print_every' iterations
        self.user_based = user_based    # user-based or item-based
        # number of users, items, and ratings. Add 1 since id starts from 0
        self.n_users = np.max(Y_data[:, 0]) + 1 
        self.n_items = np.max(Y_data[:, 1]) + 1
        self.n_ratings = Y_data.shape[0]
        
        if Xinit is None: # new
            self.X = np.random.randn(self.n_items, K)
        else: # or from saved data
            self.X = Xinit 
        
        if Winit is None: # new
            self.W = np.random.randn(K, self.n_users)
        else: # from saved data
            self.W = Winit
            
        self.Y_normalize = self.Y_raw_data.copy()   # normalized data, update later in normalized_Y function

    def normalize_Y(self):
        if self.user_based == 1:
            user_col = 0
            n_objects = self.n_users
        else:  # if we want to normalize based on item, just switch first two columns of data
            user_col = 1
            n_objects = self.n_items

        users = self.Y_raw_data[:, user_col] 
        self.mu = np.zeros((n_objects,))   # average rating of each user before normalization
        for n in range(n_objects):
            ids = np.where(users == n)[0]       # row indices of rating done by user n
            ratings = self.Y_normalize[ids, 2]  # and the corresponding ratings 
            self.mu[n] = np.mean(ratings) 
            if np.isnan(self.mu[n]) == True:
                self.mu[n] = 0 # avoid empty array and nan value
            self.Y_normalize[ids, 2] = ratings - self.mu[n] # normalize
    
    def loss(self):
        L = 0 
        for i in range(self.n_ratings):
            # user, item, rating
            n, m, rate = int(self.Y_normalize[i, 0]), int(self.Y_normalize[i, 1]), self.Y_normalize[i, 2]
            L += 0.5 * (self.X[m, :].dot(self.W[:, n]) - rate) ** 2         # 1/2 (r - item * user)^2
        
        L /= self.n_ratings  # take average
        L += 0.5 * self.lam * (np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro'))       # regularization
        return L
    
    def get_items_rated_by_user(self, user_id):
        """
        get all items rated by user_id and the corresponding ratings
        """
        ids = np.where(self.Y_normalize[:,0] == user_id)[0] 
        item_ids = self.Y_normalize[ids, 1]
        ratings = self.Y_normalize[ids, 2]
        return (item_ids, ratings)
        
        
    def get_users_who_rate_item(self, item_id):
        """
        get all users rated item_id and the corresponding ratings
        """
        ids = np.where(self.Y_normalize[:,1] == item_id)[0] 
        user_ids = self.Y_normalize[ids, 0]
        ratings = self.Y_normalize[ids, 2]
        return (user_ids, ratings)
    
    def updateX(self):
        for m in range(self.n_items):
            user_ids, rate = self.get_users_who_rate_item(m)
            Wm = self.W[:, user_ids]
            grad_xm = (self.X[m, :].dot(Wm) - rate).dot(Wm.T)/self.n_ratings + self.lam * self.X[m, :]    # gradient
            self.X[m, :] -= self.learning_rate * grad_xm.reshape((self.K,))
    
    def updateW(self):
        for n in range(self.n_users):
            item_ids, rate = self.get_items_rated_by_user(n)
            Xn = self.X[item_ids, :]
            grad_wn = Xn.T.dot(Xn.dot(self.W[:, n]) - rate)/self.n_ratings + self.lam * self.W[:, n] # gradient
            self.W[:, n] -= self.learning_rate * grad_wn.reshape((self.K,))
    
    def fit(self):
        self.normalize_Y()
        for it in range(1,self.max_iter+1):
            self.updateX()
            self.updateW()
            if it % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y_raw_data)
                print('iter =', it, ', loss =', self.loss(), ', RMSE train =', rmse_train)
    
    def pred(self, u, i):
        """ 
        predict the rating of user u for item i
        """
        bias = (self.mu[u] if self.user_based == 1 else self.mu[i])
        pred = self.X[i, :].dot(self.W[:, u]) + bias 
        # keep results in range [0, 5]
        pred = max(0, pred)
        pred = min(5, pred)
        return pred 
        
    
    def pred_for_user(self, user_id):
        """
        predict ratings one user give all unrated items
        """
        ids = np.where(self.Y_normalize[:, 0] == user_id)[0]
        items_rated_by_u = self.Y_normalize[ids, 1].tolist()              
        
        y_pred = self.X.dot(self.W[:, user_id]) + self.mu[user_id]
        predicted_ratings = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                predicted_ratings.append((i, y_pred[i]))
        
        return predicted_ratings
    
    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0 # squared error
        for n in range(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2]) ** 2 

        RMSE = np.sqrt(SE/n_tests)
        return RMSE