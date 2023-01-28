import numpy as np

# naive way to calculate loss and grad
def svm_loss_naive(W, X, y, reg):
    d, C = W.shape 
    _, N = X.shape 
    
    loss = 0 
    dW = np.zeros_like(W)   # Return array of zeros with same shape and type as W
    for n in range(N):
        xn = X[:, n]
        score = W.T.dot(xn)
        for j in range(C):
            if j == y[n]:
                continue 
            margin = 1 - score[y[n]] + score[j]
            if margin > 0:
                loss += margin 
                dW[:, j] += xn 
                dW[:, y[n]] -= xn
    
    loss /= N 
    loss += 0.5 * reg * np.sum(W * W) # regularization
    
    dW /= N 
    dW += reg * W # gradient off regularization 
    return loss, dW

# more efficient way to compute loss and grad
def svm_loss_vectorized(W, X, y, reg):
    d, C = W.shape 
    _, N = X.shape 
    loss = 0 
    dW = np.zeros_like(W)
    
    Z = W.T.dot(X)     
    
    correct_class_score = np.choose(y, Z).reshape(N,1).T     
    margins = np.maximum(0, Z - correct_class_score + 1) 
    margins[y, np.arange(margins.shape[1])] = 0
    loss = np.sum(margins, axis = (0, 1))
    loss /= N 
    loss += 0.5 * reg * np.sum(W * W)
    
    F = (margins > 0).astype(int)
    F[y, np.arange(F.shape[1])] = np.sum(-F, axis = 0)
    dW = X.dot(F.T)/N + reg*W
    return loss, dW

# Mini-batch gradient descent
def multi_svm_GD(X, y, Winit, reg=1e-4, lr=1e-7, batch_size = 100, num_iters = 1000, print_every = 100):
    W = Winit 
    loss_history = np.zeros((num_iters))
    for it in range(num_iters):
        # randomly pick a batch of X
        idx = np.random.choice(X.shape[1], batch_size)
        X_batch = X[:, idx]
        y_batch = y[idx]

        loss_history[it], dW = svm_loss_vectorized(W, X_batch, y_batch, reg)
        W -= lr*dW 
        if print_every == False:
            continue
        if it % print_every == 1:
            print(f'iter {it}/{num_iters}, loss = {loss_history[it]}')

    return W, loss_history 

def predict(X, W):
    """
    N: data points
    D: number of features
    C: number of classes
    X: N x D data matrix
    W: D x C weight matrix
    """
    Z = W.T.dot(X.T)
    y_pred = np.argmax(Z, axis = 0)
    return y_pred
