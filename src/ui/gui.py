import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import numpy as np
from core.knapsack_problem import KnapsackProblem
from core.genetic_algorithm import GeneticAlgorithm
from core.dataset_generator import generate_knapsack_dataset
from utils.plot_utils import embed_plot
from matplotlib.figure import Figure
import os


class KnapsackGAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🧬 Genetic Algorithm – 0/1 Knapsack Problem")
        self.geometry("1200x700")

        self.dataset_path = os.path.join("data", "examples")
        self.problem = None

        self.create_menu()
        self.create_widgets()

    # -----------------------------
    # Menu Bar
    # -----------------------------
    def create_menu(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="🧮 New Generated Problem", command=self.show_generated_problem)
        file_menu.add_separator()

        example_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="📦 Load Example", menu=example_menu)
        example_menu.add_command(label="Groceries", command=lambda: self.load_example("groceries.json"))
        example_menu.add_command(label="Household", command=lambda: self.load_example("household.json"))
        example_menu.add_command(label="Magic Items", command=lambda: self.load_example("magic_items.json"))
        example_menu.add_command(label="Tech Tools", command=lambda: self.load_example("tech_tools.json"))

        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        # Help Menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def show_about(self):
        messagebox.showinfo(
            "About",
            "Genetic Algorithm – Knapsack Problem\n\n"
            "Created with ❤️ using Python, NumPy, and Tkinter.\n"
            "Features: custom datasets, live fitness plots, and flexible GA parameters."
        )

    # -----------------------------
    # Main Layout
    # -----------------------------
    def create_widgets(self):
        self.left_frame = ttk.Frame(self, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.plot_frame = ttk.Frame(self, padding=10)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_parameter_panel()

    def _build_parameter_panel(self):
        lf = self.left_frame
        for widget in lf.winfo_children():
            widget.destroy()

        ttk.Label(lf, text="Dataset Parameters", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 5))
        self.entry_items = self._add_labeled_entry(lf, "Items (n_items):", 60)
        self.entry_capacity = self._add_labeled_entry(lf, "Capacity ratio:", 0.3)
        self.entry_heavy = self._add_labeled_entry(lf, "Heavy items:", 10)
        self.entry_efficient = self._add_labeled_entry(lf, "Efficient items:", 10)
        self.entry_deceptive = self._add_labeled_entry(lf, "Deceptive items:", 10)
        self.entry_seed = self._add_labeled_entry(lf, "Seed:", 42)

        ttk.Separator(lf, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(lf, text="Genetic Algorithm", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 5))
        self.entry_pop = self._add_labeled_entry(lf, "Population size:", 80)
        self.entry_gen = self._add_labeled_entry(lf, "Generations:", 150)
        self.entry_cross = self._add_labeled_entry(lf, "Crossover rate:", 0.8)
        self.entry_mut = self._add_labeled_entry(lf, "Mutation rate:", 0.05)
        self.entry_tour = self._add_labeled_entry(lf, "Tournament size:", 3)

        ttk.Separator(lf, orient="horizontal").pack(fill="x", pady=10)

        ttk.Button(lf, text="🚀 Run Genetic Algorithm", command=self.run_ga).pack(fill="x", pady=5)
        self.info_label = ttk.Label(lf, text="", wraplength=250, justify="left")
        self.info_label.pack(fill="x", pady=5)

    def _add_labeled_entry(self, parent, label, default):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text=label, width=18).pack(side=tk.LEFT)
        var = tk.StringVar(value=str(default))
        ttk.Entry(frame, textvariable=var, width=10).pack(side=tk.RIGHT)
        return var

    # -----------------------------
    # Menu Actions
    # -----------------------------
    def show_generated_problem(self):
        """Reset UI for a generated dataset."""
        self.problem = None
        self._build_parameter_panel()
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        self.info_label.config(text="🧮 Ready to generate a new random problem.")

    def load_example(self, filename):
        try:
            filepath = os.path.join(self.dataset_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            values = [item["value"] for item in data]
            weights = [item["weight"] for item in data]
            names = [item["name"] for item in data]
            capacity = sum(weights) * 0.4  # adjustable default

            self.problem = KnapsackProblem(values, weights, capacity, names)
            self.info_label.config(text=f"📦 Loaded dataset: {filename}\nItems: {len(values)}, Capacity: {capacity:.2f}")

            for widget in self.plot_frame.winfo_children():
                widget.destroy()

        except Exception as e:
            messagebox.showerror("Error loading file", str(e))

    # -----------------------------
    # Run GA
    # -----------------------------
    def run_ga(self):
        try:
            if self.problem is None:
                # Generate a problem dynamically
                n_items = int(self.entry_items.get())
                cap_ratio = float(self.entry_capacity.get())
                n_heavy = int(self.entry_heavy.get())
                n_eff = int(self.entry_efficient.get())
                n_dec = int(self.entry_deceptive.get())
                seed = int(self.entry_seed.get())

                values, weights, capacity, _ = generate_knapsack_dataset(
                    n_items=n_items, capacity_ratio=cap_ratio,
                    n_heavy=n_heavy, n_efficient=n_eff, n_deceptive=n_dec, seed=seed
                )
                self.problem = KnapsackProblem(values, weights, capacity)

            # GA parameters
            pop_size = int(self.entry_pop.get())
            generations = int(self.entry_gen.get())
            cross = float(self.entry_cross.get())
            mut = float(self.entry_mut.get())
            tour = int(self.entry_tour.get())

            ga = GeneticAlgorithm(
                self.problem, pop_size, generations,
                crossover_rate=cross, mutation_rate=mut,
                tournament_size=tour, elitism=True
            )

            best_ind, best_val, best_wt = ga.train(verbose=False)
            self.info_label.config(text=f"✅ Best Value: {best_val:.2f}\n⚖️ Weight: {best_wt:.2f} / {self.problem.capacity:.2f}")

            self._plot_results(ga)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _plot_results(self, ga):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(8, 4), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.plot(ga.history["best"], label="Best", color="green")
        ax1.plot(ga.history["avg"], label="Average", linestyle="--", color="gray")
        ax1.set_title("Fitness Evolution")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        ax1.legend()
        ax1.grid(True)

        selected = ga.best_individual.astype(bool)
        v, w = ga.problem.values, ga.problem.weights
        names = getattr(ga.problem, "names", None)

        ax2.scatter(w, v, c="gray", label="Unselected")
        ax2.scatter(w[selected], v[selected], c="green", label="Selected")

        # Add labels if names are available
        if names is not None:
            for i, name in enumerate(names):
                color = "green" if selected[i] else "gray"
                ax2.text(w[i] + 0.1, v[i] + 0.1, name, fontsize=8, color=color, alpha=0.8)

        ax2.set_title(f"Selected Items\nV={ga.best_value:.1f}, W={ga.best_weight:.1f}")
        ax2.set_xlabel("Weight")
        ax2.set_ylabel("Value")
        ax2.legend()
        ax2.grid(True)


        embed_plot(self.plot_frame, fig)
