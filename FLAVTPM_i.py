import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from ortools.linear_solver import pywraplp

np.set_printoptions(linewidth=1000)


class ProductionProblemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Production Problem Solver (Task 6)")

        self.create_widgets()

    def create_widgets(self):
        self.entries = {}
        labels = [
            "Number of aggregated products:",
            "Number of production factors:",
            "Number of assigned products:",
            "Production matrix (row-wise, separated by commas):",
            "Y assigned values (separated by commas):",
            "B values (separated by commas):",
            "C values (separated by commas):",
            "F values (separated by commas):",
            "Priorities values (separated by commas):",
            "Directive terms values (separated by commas):",
            "T_0 values (separated by commas):",
            "Alpha values (separated by commas):"
        ]

        for i, label in enumerate(labels):
            ttk.Label(self.root, text=label).grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
            self.entries[label] = ttk.Entry(self.root, width=50)
            self.entries[label].grid(row=i, column=1, padx=10, pady=5)

        ttk.Button(self.root, text="Solve", command=self.solve).grid(row=len(labels), column=0, columnspan=2, pady=10)
        ttk.Button(self.root, text="Load from file", command=self.load_from_file).grid(row=len(labels) + 1, column=0, columnspan=2, pady=10)
        self.output_text = tk.Text(self.root, wrap=tk.WORD, width=100, height=20)
        self.output_text.grid(row=len(labels) + 2, column=0, columnspan=2, pady=10)

    def get_input(self, label):
        return self.entries[label].get()

    def solve(self):
        try:
            num_aggregated_products = int(self.get_input("Number of aggregated products:"))
            num_production_factors = int(self.get_input("Number of production factors:"))
            num_assigned_products = int(self.get_input("Number of assigned products:"))

            production_matrix = np.array(
                list(map(float, self.get_input("Production matrix (row-wise, separated by commas):").split(','))),
                dtype=float).reshape((num_aggregated_products, num_production_factors))
            y_assigned = list(map(float, self.get_input("Y assigned values (separated by commas):").split(',')))
            b = list(map(float, self.get_input("B values (separated by commas):").split(',')))
            c = list(map(float, self.get_input("C values (separated by commas):").split(',')))
            f = list(map(float, self.get_input("F values (separated by commas):").split(',')))
            priorities = list(map(float, self.get_input("Priorities values (separated by commas):").split(',')))
            directive_terms = list(map(float, self.get_input("Directive terms values (separated by commas):").split(',')))
            t_0 = list(map(float, self.get_input("T_0 values (separated by commas):").split(',')))
            alpha = list(map(float, self.get_input("Alpha values (separated by commas):").split(',')))

            production_data = (
                production_matrix, y_assigned, b, c, f, priorities, directive_terms, t_0, alpha)

            y_solution, z_solution, objective_value = solve_production_problem(production_data)
            t_0 = production_data[7]
            alpha = production_data[8]
            policy_deadlines = [t_0[i] + alpha[i] * y_solution[i] for i in range(num_assigned_products)]
            completion_dates = [z_solution[i] for i in range(num_assigned_products)]
            differences = [policy_deadlines[i] - completion_dates[i] for i in range(num_assigned_products)]

            output = f"Objective Value: {objective_value}\n\n"
            output += "Y Solution: " + ', '.join(map(str, y_solution)) + "\n\n"
            output += "Z Solution: " + ', '.join(map(str, z_solution)) + "\n\n"
            output += "Policy Deadlines: " + ', '.join(map(str, policy_deadlines)) + "\n\n"
            output += "Completion Dates: " + ', '.join(map(str, completion_dates)) + "\n\n"
            output += "Differences: " + ', '.join(map(str, differences)) + "\n\n"

            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, output)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_from_file(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            if not file_path:
                return

            with open(file_path, 'r') as file:
                lines = file.readlines()

            self.entries["Number of aggregated products:"].insert(0, lines[0].strip())
            self.entries["Number of production factors:"].insert(0, lines[1].strip())
            self.entries["Number of assigned products:"].insert(0, lines[2].strip())
            self.entries["Production matrix (row-wise, separated by commas):"].insert(0, lines[3].strip())
            self.entries["Y assigned values (separated by commas):"].insert(0, lines[4].strip())
            self.entries["B values (separated by commas):"].insert(0, lines[5].strip())
            self.entries["C values (separated by commas):"].insert(0, lines[6].strip())
            self.entries["F values (separated by commas):"].insert(0, lines[7].strip())
            self.entries["Priorities values (separated by commas):"].insert(0, lines[8].strip())
            self.entries["Directive terms values (separated by commas):"].insert(0, lines[9].strip())
            self.entries["T_0 values (separated by commas):"].insert(0, lines[10].strip())
            self.entries["Alpha values (separated by commas):"].insert(0, lines[11].strip())

        except Exception as e:
            messagebox.showerror("Error", str(e))


def solve_production_problem(production_data):
    """Defines and solves the linear programming problem for production optimization."""
    production_matrix, y_assigned, b, c, f, priorities, directive_terms, t_0, alpha = production_data

    lp_solver = pywraplp.Solver.CreateSolver("GLOP")

    # Define Variables
    y = [lp_solver.NumVar(0, lp_solver.infinity(), f"y_{i}") for i in range(len(c))]
    z = [lp_solver.NumVar(0, lp_solver.infinity(), f"z_{i}") for i in range(len(f))]

    # Define Constraints
    for i in range(len(production_matrix)):
        lp_solver.Add(sum(production_matrix[i][j] * y[j] for j in range(len(c))) <= b[i])
    for i in range(len(f)):
        lp_solver.Add(z[i] >= 0)
        lp_solver.Add((t_0[i] + alpha[i] * y[i]) - z[i] <= directive_terms[i])
    for i in range(len(f)):
        lp_solver.Add(y[i] >= y_assigned[i])

    # Define Objective Function
    objective = lp_solver.Objective()
    for l in range(len(c)):
        objective.SetCoefficient(y[l], c[l] * priorities[l])
    for i in range(len(f)):
        objective.SetCoefficient(z[i], -f[i])
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(len(c))], [z[i].solution_value() for i in range(len(f))], objective.Value()


if __name__ == "__main__":
    root = tk.Tk()
    app = ProductionProblemApp(root)
    root.mainloop()
