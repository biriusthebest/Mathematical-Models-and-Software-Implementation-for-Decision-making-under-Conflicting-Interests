import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from ortools.linear_solver import pywraplp

np.set_printoptions(linewidth=1000)

class ProductionProblemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Production Problem Solver (Task 9)")

        self.create_widgets()

    def create_widgets(self):
        self.entries = {}
        labels = [
            "Number of aggregated products:",
            "Number of production factors:",
            "Number of assigned products:",
            "L value:",
            "M_L value:",
            "Production matrix (row-wise, separated by commas):",
            "Y assigned values (separated by commas):",
            "B values (separated by commas):",
            "C_L values (separated by commas):",
            "F values (separated by commas):",
            "Priorities values (separated by commas):",
            "Directive terms values (separated by commas):",
            "T_0 values (separated by commas):",
            "Alpha values (separated by commas):",
            "Omega values (separated by commas):",
            "P_L values (separated by commas):"
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
            L = int(self.get_input("L value:"))
            M_L = int(self.get_input("M_L value:"))

            production_matrix = np.array(
                list(map(float, self.get_input("Production matrix (row-wise, separated by commas):").split(','))),
                dtype=float).reshape((num_aggregated_products, num_production_factors))
            y_assigned = list(map(float, self.get_input("Y assigned values (separated by commas):").split(',')))
            b = list(map(float, self.get_input("B values (separated by commas):").split(',')))

            c_l_values = list(map(float, self.get_input("C_L values (separated by commas):").split(',')))
            C_L = np.array(c_l_values, dtype=float).reshape((M_L, L, num_production_factors))

            f = list(map(float, self.get_input("F values (separated by commas):").split(',')))
            priorities = list(map(float, self.get_input("Priorities values (separated by commas):").split(',')))
            directive_terms = list(map(float, self.get_input("Directive terms values (separated by commas):").split(',')))
            t_0 = list(map(float, self.get_input("T_0 values (separated by commas):").split(',')))
            alpha = list(map(float, self.get_input("Alpha values (separated by commas):").split(',')))
            omega = list(map(float, self.get_input("Omega values (separated by commas):").split(',')))
            P_L = list(map(float, self.get_input("P_L values (separated by commas):").split(',')))

            production_data = (
                num_aggregated_products, num_production_factors, num_assigned_products, L, M_L,
                production_matrix, y_assigned, b, C_L, f, priorities, directive_terms, t_0, alpha, omega, P_L
            )

            F_L_M_optimums = []
            for m in range(M_L):
                inner_optimums = []
                for l in range(L):
                    temp_test_production_data = (
                        production_matrix, y_assigned, b, C_L[m][l], f, priorities, directive_terms, t_0, alpha)
                    _, _, objective_value = find_temp_optimal_solution(temp_test_production_data,
                                                                       num_production_factors, num_assigned_products,
                                                                       num_aggregated_products)
                    inner_optimums.append(objective_value)
                F_L_M_optimums.append(inner_optimums)

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

            output += "F_L_M_optimums:\n"
            output += "\t" + "\t".join([f"{l = }" for l in range(L)]) + "\n"
            for m in range(M_L):
                output += f"{m = }" + "\t" + "\t".join([f"{F_L_M_optimums[m][l]:.2f}" for l in range(L)]) + "\n"

            output += "\nDifferences between f_optimum and f_solution:\n"
            output += "\t" + "\t".join([f"{l = }" for l in range(L)]) + "\n"
            for m in range(M_L):
                output += f"{m = }" + "\t"
                for l in range(L):
                    inner_difference = 0
                    for i in range(num_assigned_products):
                        inner_difference += C_L[m][l][i] * y_solution[i] - f[i] * z_solution[i]
                    optimum_value = F_L_M_optimums[m][l]
                    difference = optimum_value - inner_difference
                    output += f"{difference:12.2f}" + "\t"
                output += "\n"

            output += "\nMean differences:\n"
            for l in range(L):
                mean_difference = 0
                weighted_optimum_values = 0
                weighted_inner_differences = 0
                for m in range(M_L):
                    inner_difference = 0
                    for i in range(num_assigned_products):
                        inner_difference += C_L[m][l][i] * y_solution[i] - f[i] * z_solution[i]
                    optimum_value = F_L_M_optimums[m][l]
                    weighted_optimum_values += P_L[m] * optimum_value
                    weighted_inner_differences += P_L[m] * inner_difference
                mean_difference = weighted_optimum_values - weighted_inner_differences
                output += f"{l = },\t{weighted_optimum_values = :.2f},\t{weighted_inner_differences = :.2f},\t{mean_difference = :.2f}\n"

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
            self.entries["L value:"].insert(0, lines[3].strip())
            self.entries["M_L value:"].insert(0, lines[4].strip())
            self.entries["Production matrix (row-wise, separated by commas):"].insert(0, lines[5].strip())
            self.entries["Y assigned values (separated by commas):"].insert(0, lines[6].strip())
            self.entries["B values (separated by commas):"].insert(0, lines[7].strip())
            self.entries["C_L values (separated by commas):"].insert(0, lines[8].strip())
            self.entries["F values (separated by commas):"].insert(0, lines[9].strip())
            self.entries["Priorities values (separated by commas):"].insert(0, lines[10].strip())
            self.entries["Directive terms values (separated by commas):"].insert(0, lines[11].strip())
            self.entries["T_0 values (separated by commas):"].insert(0, lines[12].strip())
            self.entries["Alpha values (separated by commas):"].insert(0, lines[13].strip())
            self.entries["Omega values (separated by commas):"].insert(0, lines[14].strip())
            self.entries["P_L values (separated by commas):"].insert(0, lines[15].strip())

        except Exception as e:
            messagebox.showerror("Error", str(e))


def find_temp_optimal_solution(temp_production_data, num_production_factors, num_assigned_products, num_aggregated_products):
    """Defines and solves the linear programming problem for production optimization."""
    production_matrix, y_assigned, b, c_l_m, f, priorities, directive_terms, t_0, alpha = temp_production_data

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
    for i in range(num_production_factors):
        objective.SetCoefficient(y[i], c_l_m[i] * priorities[i])
    for i in range(num_assigned_products):
        objective.SetCoefficient(z[i], -f[i])
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(num_assigned_products)], objective.Value()


def solve_production_problem(production_data):
    """Defines and solves the linear programming problem for production optimization."""
    num_aggregated_products, num_production_factors, num_assigned_products, L, M_L, production_matrix, y_assigned, b, C_L, f, priorities, directive_terms, t_0, alpha, omega, P_L = production_data

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
    for l, o in enumerate(omega):
        for m in range(M_L):
            for i in range(num_production_factors):
                objective.SetCoefficient(y[i], C_L[m][l][i] * priorities[i] * o * P_L[m])
            for i in range(num_assigned_products):
                objective.SetCoefficient(z[i], -f[i] * P_L[m])
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(num_assigned_products)], objective.Value()


if __name__ == "__main__":
    root = tk.Tk()
    app = ProductionProblemApp(root)
    root.mainloop()
