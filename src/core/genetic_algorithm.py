import numpy as np
import random
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, problem, pop_size=50, generations=150,
                 crossover_rate=0.8, mutation_rate=0.05,
                 tournament_size=3, elitism=True, seed=None):
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.history = {"best": [], "avg": []}
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def initialize_population(self):
        return np.random.randint(2, size=(self.pop_size, self.problem.n_items))

    def tournament_selection(self, pop, fitnesses):
        selected = []
        for _ in range(self.pop_size):
            idx = np.random.choice(range(self.pop_size), self.tournament_size, replace=False)
            winner = pop[idx[np.argmax(fitnesses[idx])]]
            selected.append(winner)
        return np.array(selected)

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.problem.n_items - 1)
            c1 = np.concatenate((parent1[:point], parent2[point:]))
            c2 = np.concatenate((parent2[:point], parent1[point:]))
            return c1, c2
        return parent1.copy(), parent2.copy()

    def mutate(self, individual):
        for i in range(self.problem.n_items):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    def evolve(self, pop):
        fitnesses = np.array([self.problem.fitness(ind) for ind in pop])
        next_pop = []

        if self.elitism:
            elite = pop[np.argmax(fitnesses)].copy()
            next_pop.append(elite)

        selected = self.tournament_selection(pop, fitnesses)
        for i in range(0, self.pop_size - len(next_pop), 2):
            p1, p2 = selected[i], selected[i + 1]
            c1, c2 = self.crossover(p1, p2)
            next_pop.extend([self.mutate(c1), self.mutate(c2)])

        return np.array(next_pop[:self.pop_size]), fitnesses

    def train(self, verbose=True):
        pop = self.initialize_population()
        for gen in range(self.generations):
            pop, fitnesses = self.evolve(pop)
            best, avg = np.max(fitnesses), np.mean(fitnesses)
            self.history["best"].append(best)
            self.history["avg"].append(avg)
            if verbose and gen % 10 == 0:
                print(f"Gen {gen:03d} | Best: {best:.2f} | Avg: {avg:.2f}")

        final_fit = np.array([self.problem.fitness(ind) for ind in pop])
        best_idx = np.argmax(final_fit)
        self.best_individual = pop[best_idx]
        self.best_value = np.sum(self.best_individual * self.problem.values)
        self.best_weight = np.sum(self.best_individual * self.problem.weights)
        return self.best_individual, self.best_value, self.best_weight

    def plot_fitness(self, ax=None):
        if ax is None:
            plt.figure(figsize=(8, 5))
            ax = plt.gca()
        ax.plot(self.history["best"], label="Best Fitness")
        ax.plot(self.history["avg"], label="Average Fitness", linestyle="--")
        ax.set_title("GA – Fitness Evolution")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.legend()
        ax.grid(True)
        if ax is plt.gca():
            plt.show()

    def plot_items(self):
        selected = self.best_individual.astype(bool)
        v, w = self.problem.values, self.problem.weights
        plt.figure(figsize=(7, 5))
        plt.scatter(w, v, color='gray', label='Unselected')
        plt.scatter(w[selected], v[selected], color='green', label='Selected')
        plt.title(f"Selected Items (Total W={self.best_weight}, V={self.best_value})")
        plt.xlabel("Weight")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()
