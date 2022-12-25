import pandas as pd 
import numpy as np
from Low_rank_Matrix_factorization import MF

class MF_bias(MF):
    def __init__(self, Y_data, K, lam = 0.1, Xinit = None, Winit = None, learning_rate = 0.5, 
                max_iter = 1000, print_every = 100, user_based = 1):
        MF.__init__(self, Y_data, K, lam, Xinit, Winit, learning_rate, max_iter, print_every, user_based)
        
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        self.global_bias = 0

    def loss(self):
        L = 0 
        for i in range(self.n_ratings):
            # user, item, rating
            n, m, rate = int(self.Y_normalize[i, 0]), int(self.Y_normalize[i, 1]), self.Y_normalize[i, 2]
            # 1/2 (r - item * user - user_bias - item_bias - global_bias)^2
            L += 0.5 * (self.X[m, :].dot(self.W[:, n]) + self.item_bias[m] + self.user_bias[n] + self.global_bias - rate) ** 2         
        
        L /= self.n_ratings  # take average
        L += 0.5 * self.lam * (np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro') 
                + np.linalg.norm(self.user_bias) + np.linalg.norm(self.item_bias))       # regularization
        return L
    
    def updateX_n_user_bias(self):
        for m in range(self.n_items):
            user_ids, rate = self.get_users_who_rate_item(m)
            Wm = self.W[:, user_ids]
            user_bias_m = self.user_bias[user_ids]

            difference = self.X[m, :].dot(Wm) + self.item_bias[m] + user_bias_m + self.global_bias - rate
            grad_xm = difference * Wm.T / self.n_ratings + self.lam * self.X[m, :] # gradient
            grad_user = difference / self.n_ratings + self.lam * self.user_bias[m] 
            self.X[m, :] -= self.learning_rate * grad_xm.reshape((self.K,))
            self.user_bias[m] -= self.learning_rate * grad_user
    
    def updateW_n_item_bias(self):
        for n in range(self.n_users):
            item_ids, rate = self.get_items_rated_by_user(n)
            Xn = self.X[item_ids, :]
            item_bias_n = self.item_bias[item_ids]

            difference = Xn.dot(self.W[:,n]) + item_bias_n + self.user_bias[n] + self.global_bias - rate
            grad_wn = difference * Xn.T / self.n_ratings + self.lam * self.W[:, n] # gradient
            grad_item = difference / self.n_ratings + self.lam * self.item_bias[n]
            self.W[:, n] -= self.learning_rate * grad_wn.reshape((self.K,))
            self.item_bias[n] -= self.learning_rate * grad_item
     
    def pred(self, u, i):
        """ 
        predict the rating of user u for item i
        """
        bias = (self.mu[u] if self.user_based == 1 else self.mu[i])
        pred = self.X[i, :].dot(self.W[:, u]) + bias + self.user_bias[u] + self.item_bias[i] + self.global_bias
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
        
        y_pred = self.X.dot(self.W[:, user_id]) + self.mu[user_id] + self.user_bias[user_id] + self.item_bias + self.global_bias
        predicted_ratings = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                predicted_ratings.append((i, y_pred[i]))
        
        return predicted_ratings