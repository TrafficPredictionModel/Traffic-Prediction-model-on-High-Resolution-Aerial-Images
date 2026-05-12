import time
import numpy as np

def SCOA(X,obj_func, lb,ub,  max_iter,):
    pop_size,D = X.shape
    seed = None
    verbose = False
    if seed is not None:
        np.random.seed(seed)
    F = np.array([obj_func(x) for x in X])
    best_idx = np.argmin(F)
    best, best_f = X[best_idx].copy(), F[best_idx]
    history = [best_f]

    ct=time.time()
    for t in range(1, max_iter + 1):
        # Phase 1: Exploration
        for i in range(pop_size):
            r = np.random.rand(D)    # vector in [0,1]
            I = np.random.choice([1, 2])
            X_p1 = X[i] + r * (best - I * X[i])
            X_p1 = np.clip(X_p1, lb, ub)
            f_p1 = obj_func(X_p1)
            if f_p1 <= F[i]:
                X[i], F[i] = X_p1, f_p1
                if f_p1 < best_f:
                    best, best_f = X_p1.copy(), f_p1

        # Phase 2: Exploitation
        for i in range(pop_size):
            w_self = (max_iter - t) / max_iter
            w_best = t / max_iter
            X_p2 = w_self * X[i] + w_best * best
            X_p2 = np.clip(X_p2, lb, ub)
            f_p2 = obj_func(X_p2)
            if f_p2 <= F[i]:
                X[i], F[i] = X_p2, f_p2
                if f_p2 < best_f:
                    best, best_f = X_p2.copy(), f_p2

        history.append(best_f)
    ct=time.time()-ct

    return best_f, history,best,ct

