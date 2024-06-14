import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from ortools.linear_solver import pywraplp

np.set_printoptions(linewidth=1000)

class ProductionProblemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Production Problem Solver (Task 7)")

        self.create_widgets()

    def create_widgets(self):
        self.entries = {}
        labels = [
            "Number of aggregated products:",
            "Number of production factors:",
            "Number of assigned products:",
            "M value:",
            "Production matrix (row-wise, separated by commas):",
            "Y assigned values (separated by commas):",
            "B values (separated by commas):",
            "C matrix values (separated by commas):",
            "P_m values (separated by commas):",
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
            M = int(self.get_input("M value:"))

            production_matrix = np.array(
                list(map(float, self.get_input("Production matrix (row-wise, separated by commas):").split(','))),
                dtype=float).reshape((num_aggregated_products, num_production_factors))
            y_assigned = list(map(float, self.get_input("Y assigned values (separated by commas):").split(',')))
            b = list(map(float, self.get_input("B values (separated by commas):").split(',')))

            c_values = list(map(float, self.get_input("C matrix values (separated by commas):").split(',')))
            C = np.array(c_values, dtype=float).reshape((M, num_production_factors))

            P_m = list(map(float, self.get_input("P_m values (separated by commas):").split(',')))
            f = list(map(float, self.get_input("F values (separated by commas):").split(',')))
            priorities = list(map(float, self.get_input("Priorities values (separated by commas):").split(',')))
            directive_terms = list(map(float, self.get_input("Directive terms values (separated by commas):").split(',')))
            t_0 = list(map(float, self.get_input("T_0 values (separated by commas):").split(',')))
            alpha = list(map(float, self.get_input("Alpha values (separated by commas):").split(',')))

            production_data = (
                num_aggregated_products, num_production_factors, num_assigned_products, M,
                production_matrix, y_assigned, b, C, P_m, f, priorities, directive_terms, t_0, alpha
            )

            F_optimums = [find_temp_optimal_solution(production_data, m)[2] for m in range(M)]
            y_solution, z_solution, objective_value = solve_production_problem(production_data)
            policy_deadlines = [t_0[i] + alpha[i] * y_solution[i] for i in range(num_assigned_products)]
            completion_dates = [z_solution[i] for i in range(num_assigned_products)]
            differences = [policy_deadlines[i] - completion_dates[i] for i in range(num_assigned_products)]

            output = f"Objective Value: {objective_value}\n\n"
            output += "Y Solution: " + ', '.join(map(str, y_solution)) + "\n\n"
            output += "Z Solution: " + ', '.join(map(str, z_solution)) + "\n\n"
            output += "Policy Deadlines: " + ', '.join(map(str, policy_deadlines)) + "\n\n"
            output += "Completion Dates: " + ', '.join(map(str, completion_dates)) + "\n\n"
            output += "Differences: " + ', '.join(map(str, differences)) + "\n\n"
            output += "Differences between f_optimum and f_solution:\n"

            mean = 0
            for m in range(M):
                c_l_m = C[m]
                f_solution = sum([c_l_m[i] * y_solution[i] for i in range(num_production_factors)]) - sum(
                    [f[i] * z_solution[i] for i in range(num_assigned_products)])
                difference = F_optimums[m] - f_solution
                output += f"{m = }, {F_optimums[m] = :.2f}, {f_solution = :.2f}, {difference = :.2f}\n"
                mean += P_m[m] * difference
            output += f"Mean difference: {mean:.2f}\n"

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
            self.entries["M value:"].insert(0, lines[3].strip())
            self.entries["Production matrix (row-wise, separated by commas):"].insert(0, lines[4].strip())
            self.entries["Y assigned values (separated by commas):"].insert(0, lines[5].strip())
            self.entries["B values (separated by commas):"].insert(0, lines[6].strip())
            self.entries["C matrix values (separated by commas):"].insert(0, lines[7].strip())
            self.entries["P_m values (separated by commas):"].insert(0, lines[8].strip())
            self.entries["F values (separated by commas):"].insert(0, lines[9].strip())
            self.entries["Priorities values (separated by commas):"].insert(0, lines[10].strip())
            self.entries["Directive terms values (separated by commas):"].insert(0, lines[11].strip())
            self.entries["T_0 values (separated by commas):"].insert(0, lines[12].strip())
            self.entries["Alpha values (separated by commas):"].insert(0, lines[13].strip())

        except Exception as e:
            messagebox.showerror("Error", str(e))


def find_temp_optimal_solution(production_data, m):
    """Finds the temporary optimal solution for the given production data."""
    num_aggregated_products, num_production_factors, num_assigned_products, _, production_matrix, y_assigned, b, c, _, f, priorities, directive_terms, t_0, alpha = production_data

    lp_solver = pywraplp.Solver.CreateSolver("GLOP")

    # Define Variables
    y = [lp_solver.NumVar(0, lp_solver.infinity(), f"y_{i}") for i in range(num_production_factors)]
    z = [lp_solver.NumVar(0, lp_solver.infinity(), f"z_{i}") for i in range(num_assigned_products)]

    # Define Constraints
    for i in range(num_aggregated_products):
        lp_solver.Add(sum(production_matrix[i][j] * y[j] for j in range(num_production_factors)) <= b[i])
    for i in range(num_assigned_products):
        lp_solver.Add(z[i] >= 0)
        lp_solver.Add((t_0[i] + alpha[i] * y[i]) - z[i] <= directive_terms[i])
    for i in range(num_assigned_products):
        lp_solver.Add(y[i] >= y_assigned[i])

    # Define Objective Function
    objective = lp_solver.Objective()
    for l in range(num_production_factors):
        objective.SetCoefficient(y[l], c[m][l] * priorities[l])
    for i in range(num_assigned_products):
        objective.SetCoefficient(z[i], -f[i])
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
        num_assigned_products)], objective.Value()


def solve_production_problem(production_data):
    """Defines and solves the linear programming problem for production optimization."""
    num_aggregated_products, num_production_factors, num_assigned_products, M, production_matrix, y_assigned, b, c, P_m, f, priorities, directive_terms, t_0, alpha = production_data

    lp_solver = pywraplp.Solver.CreateSolver("GLOP")

    # Define Variables
    y = [lp_solver.NumVar(0, lp_solver.infinity(), f"y_{i}") for i in range(num_production_factors)]
    z = [lp_solver.NumVar(0, lp_solver.infinity(), f"z_{i}") for i in range(num_assigned_products)]

    # Define Constraints
    for i in range(num_aggregated_products):
        lp_solver.Add(sum(production_matrix[i][j] * y[j] for j in range(num_production_factors)) <= b[i])
    for i in range(num_assigned_products):
        lp_solver.Add(z[i] >= 0)
        lp_solver.Add((t_0[i] + alpha[i] * y[i]) - z[i] <= directive_terms[i])
    for i in range(num_assigned_products):
        lp_solver.Add(y[i] >= y_assigned[i])

    # Define Objective Function
    objective = lp_solver.Objective()
    for i, p_m in enumerate(P_m):
        for l, p in enumerate(priorities):
            objective.SetCoefficient(y[l], p_m * c[i][l] * p)
        for l in range(num_assigned_products):
            objective.SetCoefficient(z[l], -p_m * f[l])
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
        num_assigned_products)], objective.Value()


if __name__ == "__main__":
    root = tk.Tk()
    app = ProductionProblemApp(root)
    root.mainloop()
