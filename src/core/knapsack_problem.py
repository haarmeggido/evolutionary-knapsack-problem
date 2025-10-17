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
        if total_weight <= self.capacity:
            return total_value
        penalty = 10 * (total_weight - self.capacity)
        return total_value - penalty
