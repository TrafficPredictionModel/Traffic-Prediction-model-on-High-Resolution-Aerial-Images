import numpy as np
import time

# Carpet Weaver Optimization
def CWO(initsol, fobj, xmin, xmax, Max_iter):
    N, dim = initsol.shape  # Number of carpets (population size) and dimensions

    # Initialize population
    carpets = initsol.copy()
    fitness = np.array([fobj(carpets[i]) for i in range(N)])

    # Initialize global best
    best_idx = np.argmin(fitness)
    gBest = carpets[best_idx].copy()
    gBestScore = fitness[best_idx]

    # Convergence curve
    Convergence_curve = np.zeros(Max_iter)
    Convergence_curve[0] = gBestScore

    start_time = time.time()

    for t in range(1, Max_iter):
        r = np.random.rand()  # Random number for updates

        # Phase 1: Exploration (Carpet weaving based on the given pattern)
        pattern = xmin + np.random.rand(dim) * (xmax - xmin)  # Random weaving pattern
        for i in range(N):
            I = np.random.choice([1, 2])  # Random integer (1 or 2)
            new_position = carpets[i] + (1 - 2 * np.random.rand(dim)) * (pattern - I * carpets[i])
            new_position = np.clip(new_position, xmin, xmax)
            new_fitness = fobj(new_position)

            # Update if better
            if new_fitness[i] < fitness[i]:
                carpets[i] = new_position[i]
                fitness[i] = new_fitness[i]

        # Phase 2: Exploitation (Creative changes to the design)
        for i in range(N):
            new_position = carpets[i] * (1 + (1 - 2 * np.random.rand(dim)) / (t + 1))
            new_position = np.clip(new_position, xmin, xmax)
            new_fitness = fobj(new_position)

            # Update if better
            if new_fitness[i] < fitness[i]:
                carpets[i] = new_position[i]
                fitness[i] = new_fitness[i]

        # Update global best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < gBestScore:
            gBest = carpets[best_idx].copy()
            gBestScore = fitness[best_idx]

        Convergence_curve[t] = gBestScore

    time4 = time.time() - start_time
    return gBest, gBestScore, Convergence_curve, time4
