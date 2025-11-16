import random
import numpy as np
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    def __init__(
        self,
        problem,
        pop_size=50,
        generations=150,
        crossover_rate=0.8,
        mutation_rate=0.05,
        tournament_size=3,
        elitism=True,
        seed=None
    ):
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.cxpb = crossover_rate
        self.mutpb = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.history = {"best": [], "avg": []}

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Individual is a binary list
        self.toolbox.register(
            "attr_bool",
            lambda: random.randint(0, 1)
        )

        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_bool,
            n=self.problem.n_items
        )

        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )

        def eval_knapsack(individual):
            return (self.problem.fitness(np.array(individual)),)

        self.toolbox.register("evaluate", eval_knapsack)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        self.toolbox.register("mate", tools.cxOnePoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=self.mutpb)


    def train(self, verbose=True):
        pop = self.toolbox.population(n=self.pop_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for gen in range(self.generations):
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    self.toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate dirty individuals
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = self.toolbox.evaluate(ind)

            # Elitism
            if self.elitism:
                elite = tools.selBest(pop, 1)[0]
                offspring[0] = self.toolbox.clone(elite)

            pop = offspring

            # Track stats
            fits = [ind.fitness.values[0] for ind in pop]
            best = max(fits)
            avg = np.mean(fits)
            self.history["best"].append(best)
            self.history["avg"].append(avg)

            if verbose and gen % 10 == 0:
                print(f"Gen {gen:03d} | Best: {best:.2f} | Avg: {avg:.2f}")

        # Final evaluation
        best_ind = tools.selBest(pop, 1)[0]
        self.best_individual = np.array(best_ind)
        self.best_value = np.sum(self.best_individual * self.problem.values)
        self.best_weight = np.sum(self.best_individual * self.problem.weights)
        return self.best_individual, self.best_value, self.best_weight


    def plot_fitness(self, ax=None):
        if ax is None:
            plt.figure(figsize=(8, 5))
            ax = plt.gca()
        ax.plot(self.history["best"], label="Best Fitness")
        ax.plot(self.history["avg"], label="Average Fitness", linestyle="--")
        ax.set_title("GA – Fitness Evolution (DEAP)")
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
