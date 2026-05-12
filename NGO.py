import numpy as np
import time
def NGO(X, objective, lowerbound, upperbound, max_iterations):
    search_agents,dimensions = X.shape
    fit = objective(X)
    NGO_curve = np.zeros(max_iterations)
    ct = time.time()
    # Optimization loop
    for t in range(max_iterations):
        # Update best solution
        best_idx = np.argmin(fit)
        best = fit[best_idx]
        x_best = X[best_idx].copy()

        if t == 0 or best < fbest:
            fbest = best
            xbest = x_best.copy()

        # Update Northern Goshawks based on PHASE 1 and PHASE 2
        for i in range(search_agents):
            # Phase 1: Exploration
            I = np.round(1 + np.random.rand())
            k = np.random.randint(search_agents)
            P = X[k, :]
            F_P = fit[k]

            if fit[i] > F_P:
                X_new = X[i, :] + np.random.rand(dimensions) * (P - I * X[i, :])
            else:
                X_new = X[i, :] + np.random.rand(dimensions) * (X[i, :] - P)

            X_new = np.clip(X_new, lowerbound, upperbound)
            fit_new = objective(X_new)

            if fit_new < fit[i]:
                X[i, :] = X_new
                fit[i] = fit_new

            # Phase 2: Exploitation
            R = 0.02 * (1 - t / max_iterations)
            X_new = X[i, :] + (-R + 2 * R * np.random.rand(dimensions)) * X[i, :]
            X_new = np.clip(X_new, lowerbound, upperbound)
            fit_new = objective(X_new)

            if fit_new < fit[i]:
                X[i, :] = X_new
                fit[i] = fit_new

        # Save best score
        NGO_curve[t] = fbest
    ct = time.time()-ct
    return x_best,NGO_curve, fbest, ct
