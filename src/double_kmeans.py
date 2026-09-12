import numpy as np

class DoubleKMeans:
    def __init__(self, K, H, max_iter=100, tol=1e-4, random_state=None):
        self.K = K
        self.H = H
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, n_init=10):
        best_W = np.inf
        best_model = None

        rng = np.random.RandomState(self.random_state)

        for init_idx in range(n_init):
            model = self._fit_single(X, rng)
            if model['W'][-1] < best_W:
                best_W = model['W'][-1]
                best_model = model

        self.G_ = best_model['G']
        self.U_ = best_model['U']
        self.V_ = best_model['V']
        self.W_history_ = best_model['W']
        self.best_W_ = best_W
        return self

    def _fit_single(self, X, rng):

        N, P = X.shape
        K, H = self.K, self.H

        # Initialize U and V randomly
        U = np.zeros((N, K))
        U[np.arange(N), rng.randint(0, K, N)] = 1
        
        V = np.zeros((P, H))
        V[np.arange(P), rng.randint(0, H, P)] = 1

        # Prevent empty clusters by forcing at least one random assignment
        if np.any(U.sum(axis=0) == 0):
            empty_k = np.where(U.sum(axis=0) == 0)[0]
            for k in empty_k:
                idx = rng.randint(0, N)
                U[idx, :] = 0
                U[idx, k] = 1

        if np.any(V.sum(axis=0) == 0):
            empty_h = np.where(V.sum(axis=0) == 0)[0]
            for h in empty_h:
                idx = rng.randint(0, P)
                V[idx, :] = 0
                V[idx, h] = 1

        W_history = []

        for iteration in range(self.max_iter):
            # Step 1: Update G
            Nk = U.sum(axis=0).reshape(-1, 1) # (K, 1)
            Nh = V.sum(axis=0).reshape(-1, 1) # (H, 1)
            
            # Avoid division by zero
            Nk[Nk == 0] = 1
            Nh[Nh == 0] = 1
            
            G = (U.T @ X @ V) / (Nk @ Nh.T) # (K, H)
            
            # Step 2: Update U
            # Dist_U(i, k) = const - 2(X V G^T)_{ik} + (G^2 Nh)_{k}
            term1_U = -2 * (X @ V @ G.T) # (N, K)
            term2_U = (G**2) @ Nh # (K, 1)
            dist_U = term1_U + term2_U.T # (N, K)
            
            U_new = np.zeros((N, K))
            U_new[np.arange(N), np.argmin(dist_U, axis=1)] = 1
            
            # Step 3: Update V
            # Dist_V(j, h) = const - 2(X^T U G)_{jh} + ( (G^2)^T Nk )_{h}
            term1_V = -2 * (X.T @ U_new @ G) # (P, H)
            term2_V = (G**2).T @ Nk # (H, 1)
            dist_V = term1_V + term2_V.T # (P, H)
            
            V_new = np.zeros((P, H))
            V_new[np.arange(P), np.argmin(dist_V, axis=1)] = 1
            
            # Calculate objective W
            # W = sum_{k,h,i,j} U_{ik} V_{jh} (X_{ij} - G_{kh})^2
            # W = sum_{i,j} X_{ij}^2 - 2 sum_{k,h} G_{kh} (U^T X V)_{kh} + sum_{k,h} G_{kh}^2 Nk_k Nh_h
            term1_W = np.sum(X**2)
            term2_W = -2 * np.sum(G * (U_new.T @ X @ V_new))
            Nk_new = U_new.sum(axis=0).reshape(-1, 1)
            Nh_new = V_new.sum(axis=0).reshape(-1, 1)
            term3_W = np.sum((G**2) * (Nk_new @ Nh_new.T))
            W = term1_W + term2_W + term3_W
            W_history.append(W)
            
            # Check convergence
            if np.array_equal(U, U_new) and np.array_equal(V, V_new):
                break
                
            U = U_new
            V = V_new

        # Final G update to match final U and V
        Nk = U.sum(axis=0).reshape(-1, 1)
        Nh = V.sum(axis=0).reshape(-1, 1)
        Nk[Nk == 0] = 1
        Nh[Nh == 0] = 1
        G = (U.T @ X @ V) / (Nk @ Nh.T)

        return {'G': G, 'U': U, 'V': V, 'W': W_history}
