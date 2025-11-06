import numpy as np

class KnapsackProblem:
    def __init__(self, values, weights, capacity, names=None):
        self.values = np.array(values)
        self.weights = np.array(weights)
        self.capacity = capacity
        self.n_items = len(values)
        self.names = names

    def fitness(self, individual):
        total_value = np.sum(individual * self.values)
        total_weight = np.sum(individual * self.weights)
        penalty_factor = total_value / total_weight * 10 if total_weight > 0 else 0
        if total_weight <= self.capacity:
            return total_value
        penalty = penalty_factor * (total_weight - self.capacity)
        return total_value - penalty
