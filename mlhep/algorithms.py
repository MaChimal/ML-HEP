from typing import List, Tuple
from pydantic import BaseModel
import numpy as np

class DEParams(BaseModel):
    mut: float
    crossp: float
    popsize: int
    its: int

class OptimizationAlgorithms:
    
    def differential_evolution(self, f_x, bounds, params: DEParams):
        dimensions = len(bounds)
        pop = np.random.rand(params.popsize, dimensions)
        min_b, max_b = np.asarray(bounds).T
        diff = np.fabs(min_b - max_b)
        pop_denorm = min_b + pop * diff
        fitness = np.asarray([f_x(ind) for ind in pop_denorm])
        best_idx = np.argmin(fitness)
        best = pop_denorm[best_idx]
        for i in range(params.its):
            for j in range(params.popsize):
                idxs = [idx for idx in range(params.popsize) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + params.mut * (b - c), 0, 1)
                cross_points = np.random.rand(dimensions) < params.crossp
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimensions)] = True
                trial = np.where(cross_points, mutant, pop[j])
                trial_denorm = min_b + trial * diff
                f = f_x(trial_denorm)
                if f < fitness[j]:
                    fitness[j] = f
                    pop[j] = trial
                    if f < fitness[best_idx]:
                        best_idx = j
                        best = trial_denorm
            yield best, fitness[best_idx]
