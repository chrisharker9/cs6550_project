import numpy as np

class NMF(object):
    def __init__(self, n_components=2, alpha=0.01, max_iter=1000, tol=1e-04, verbose=False):
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
    def fit(self, X):
        
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be np.ndarray")
        if self.n_components > X.shape[1]:
            raise ValueError(f"n_components must be less than {X.shape[1]}")
        
        # Initialize Randomly
        W = np.random.uniform(low=0, high=1, size=(X.shape[0], self.n_components))
        H = np.random.uniform(low=0, high=1, size=(self.n_components, X.shape[1]))
        
        self.err_ = []
        self.err2_ = []
        err = np.finfo(np.float64).max
        err2 = np.linalg.norm(X - W @ H)
        i = 0
        while (err > self.tol) & (i < self.max_iter):
            # Update W
            
            H_new = H * ((W.T @ X)/(W.T @ (W @ H)))
            W_new = W * ((X @ H_new.T)/(W @ (H_new @ H_new.T)))
    
            err = np.linalg.norm(W_new@H_new - W @ H)
            err2 = np.linalg.norm(X - W_new@H_new)
            self.err_.append(err)
            self.err2_.append(err2)
            if self.verbose:
                print(f"i: {i}, err: {err}, err2: {err2}")
            
            i += 1
            W = W_new
            H = H_new
            
        self.W_ = W
        self.H_ = H
        self.iter_err_ = np.array(self.err_)
        self.iter_err2_ = np.array(self.err2_)
        self.final_err_ = 0.5*(np.linalg.norm(X-self.W_@self.H_))**2
        
        return self