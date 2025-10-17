# ui/gui.py
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
from core.knapsack_problem import KnapsackProblem
from core.genetic_algorithm import GeneticAlgorithm
from core.dataset_generator import generate_knapsack_dataset
from utils.plot_utils import embed_plot
from matplotlib.figure import Figure


class KnapsackGAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Genetic Algorithm – 0/1 Knapsack Problem")
        self.iconbitmap(False, 'assets/icon_fun.ico')
        self.geometry("1200x700")

        self.create_widgets()

    def create_widgets(self):
        left_frame = ttk.Frame(self, padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.plot_frame = ttk.Frame(self, padding=10)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(left_frame, text="Dataset Parameters", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0,5))

        self.entry_items = self._add_labeled_entry(left_frame, "Items (n_items):", 60)
        self.entry_capacity = self._add_labeled_entry(left_frame, "Capacity ratio:", 0.3)
        self.entry_heavy = self._add_labeled_entry(left_frame, "Heavy items:", 10)
        self.entry_efficient = self._add_labeled_entry(left_frame, "Efficient items:", 10)
        self.entry_deceptive = self._add_labeled_entry(left_frame, "Deceptive items:", 10)
        self.entry_seed = self._add_labeled_entry(left_frame, "Seed:", 42)

        ttk.Separator(left_frame, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(left_frame, text="Genetic Algorithm", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0,5))

        self.entry_pop = self._add_labeled_entry(left_frame, "Population size:", 80)
        self.entry_gen = self._add_labeled_entry(left_frame, "Generations:", 150)
        self.entry_cross = self._add_labeled_entry(left_frame, "Crossover rate:", 0.8)
        self.entry_mut = self._add_labeled_entry(left_frame, "Mutation rate:", 0.05)
        self.entry_tour = self._add_labeled_entry(left_frame, "Tournament size:", 3)

        ttk.Separator(left_frame, orient="horizontal").pack(fill="x", pady=10)

        # Run Button
        run_btn = ttk.Button(left_frame, text="🚀 Run Genetic Algorithm", command=self.run_ga)
        run_btn.pack(fill="x", pady=5)

        # Info label
        self.info_label = ttk.Label(left_frame, text="", wraplength=250, justify="left")
        self.info_label.pack(fill="x", pady=5)

    def _add_labeled_entry(self, parent, label, default):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text=label, width=18).pack(side=tk.LEFT)
        var = tk.StringVar(value=str(default))
        ttk.Entry(frame, textvariable=var, width=10).pack(side=tk.RIGHT)
        return var

    def run_ga(self):
        try:
            n_items = int(self.entry_items.get())
            capacity_ratio = float(self.entry_capacity.get())
            n_heavy = int(self.entry_heavy.get())
            n_efficient = int(self.entry_efficient.get())
            n_deceptive = int(self.entry_deceptive.get())
            seed = int(self.entry_seed.get())

            values, weights, capacity, _ = generate_knapsack_dataset(
                n_items=n_items,
                capacity_ratio=capacity_ratio,
                n_heavy=n_heavy,
                n_efficient=n_efficient,
                n_deceptive=n_deceptive,
                seed=seed
            )

            problem = KnapsackProblem(values, weights, capacity)

            pop_size = int(self.entry_pop.get())
            generations = int(self.entry_gen.get())
            cross = float(self.entry_cross.get())
            mut = float(self.entry_mut.get())
            tour = int(self.entry_tour.get())

            ga = GeneticAlgorithm(
                problem,
                pop_size=pop_size,
                generations=generations,
                crossover_rate=cross,
                mutation_rate=mut,
                tournament_size=tour,
                elitism=True,
                seed=seed
            )

            best_ind, best_val, best_wt = ga.train(verbose=False)

            self.info_label.config(text=f"✅ Best Value: {best_val:.2f}\n⚖️ Weight: {best_wt} / {capacity}")

            self._plot_results(ga)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _plot_results(self, ga):
        """Clear the plot area and embed fitness & item plots."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(8,4), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Fitness evolution
        ax1.plot(ga.history["best"], label="Best", color="green")
        ax1.plot(ga.history["avg"], label="Average", linestyle="--", color="gray")
        ax1.set_title("Fitness Evolution")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        ax1.legend()
        ax1.grid(True)

        # Selected items plot
        selected = ga.best_individual.astype(bool)
        values, weights = ga.problem.values, ga.problem.weights
        ax2.scatter(weights, values, c="gray", label="Unselected")
        ax2.scatter(weights[selected], values[selected], c="green", label="Selected")
        ax2.set_title(f"Selected Items\nTotal V={ga.best_value}, W={ga.best_weight}")
        ax2.set_xlabel("Weight")
        ax2.set_ylabel("Value")
        ax2.legend()
        ax2.grid(True)

        embed_plot(self.plot_frame, fig)
