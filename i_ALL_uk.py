import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from ortools.linear_solver import pywraplp
import solver
np.set_printoptions(linewidth=1000)


class MultiTaskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Багатозадачний додаток для вирішення виробничих задач")
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(pady=10, expand=True)

        self.create_task1_widgets()
        self.create_task2_widgets()
        self.create_task3_widgets()
        self.create_task4_widgets()

    def create_task1_widgets(self):
        self.task1_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.task1_tab, text="Задача 1")

        labels = [
            "Кількість агрегованих продуктів:",
            "Кількість факторів виробництва:",
            "Кількість призначених продуктів:",
            "Значення L:",
            "Матриця виробництва (построчно, через кому):",
            "Призначені значення Y (через кому):",
            "Значення B (через кому):",
            "Матриця значень C (через кому):",
            "Значення F (через кому):",
            "Значення пріоритетів (через кому):",
            "Значення директивних термінів (через кому):",
            "Значення T_0 (через кому):",
            "Значення альфа (через кому):",
            "Значення омега (через кому):"
        ]

        self.task1_entries = {}
        for i, label in enumerate(labels):
            ttk.Label(self.task1_tab, text=label).grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
            self.task1_entries[label] = ttk.Entry(self.task1_tab, width=50)
            self.task1_entries[label].grid(row=i, column=1, padx=10, pady=5)

        ttk.Button(self.task1_tab, text="Розв'язати задачу 1", command=self.solve_task1).grid(row=len(labels), column=0, columnspan=2, pady=10)
        ttk.Button(self.task1_tab, text="Завантажити з файлу", command=self.load_task1_from_file).grid(row=len(labels) + 1, column=0, columnspan=2, pady=10)
        self.task1_output_text = tk.Text(self.task1_tab, wrap=tk.WORD, width=100, height=20)
        self.task1_output_text.grid(row=len(labels) + 2, column=0, columnspan=2, pady=10)

    def solve_task1(self):
        try:
            num_aggregated_products = int(self.task1_entries["Кількість агрегованих продуктів:"].get())
            num_production_factors = int(self.task1_entries["Кількість факторів виробництва:"].get())
            num_assigned_products = int(self.task1_entries["Кількість призначених продуктів:"].get())
            L = int(self.task1_entries["Значення L:"].get())

            production_matrix = np.array(
                list(map(float, self.task1_entries["Матриця виробництва (построчно, через кому):"].get().split(','))),
                dtype=float).reshape((num_aggregated_products, num_production_factors))
            y_assigned = list(map(float, self.task1_entries["Призначені значення Y (через кому):"].get().split(',')))
            b = list(map(float, self.task1_entries["Значення B (через кому):"].get().split(',')))
            c = np.array(list(map(float, self.task1_entries["Матриця значень C (через кому):"].get().split(','))),
                         dtype=float).reshape((L, num_production_factors))
            f = list(map(float, self.task1_entries["Значення F (через кому):"].get().split(',')))
            priorities = list(map(float, self.task1_entries["Значення пріоритетів (через кому):"].get().split(',')))
            directive_terms = list(
                map(float, self.task1_entries["Значення директивних термінів (через кому):"].get().split(',')))
            t_0 = list(map(float, self.task1_entries["Значення T_0 (через кому):"].get().split(',')))
            alpha = list(map(float, self.task1_entries["Значення альфа (через кому):"].get().split(',')))
            omega = list(map(float, self.task1_entries["Значення омега (через кому):"].get().split(',')))

            production_data = (
                num_aggregated_products, num_production_factors, num_assigned_products, L, production_matrix, y_assigned, b,
                c, f, priorities, directive_terms, t_0, alpha, omega)

            y_solution, z_solution, objective_value = self.solve_production_problem_task1(production_data)
            t_0 = production_data[11]
            alpha = production_data[12]
            policy_deadlines = [t_0[i] + alpha[i] * y_solution[i] for i in range(num_assigned_products)]
            completion_dates = [z_solution[i] for i in range(num_assigned_products)]
            differences = [policy_deadlines[i] - completion_dates[i] for i in range(num_assigned_products)]

            output = f"Цільове значення: {objective_value}\n\n"
            output += "Рішення Y: " + ', '.join(map(str, y_solution)) + "\n\n"
            output += "Рішення Z: " + ', '.join(map(str, z_solution)) + "\n\n"
            output += "Директивні терміни: " + ', '.join(map(str, policy_deadlines)) + "\n\n"
            output += "Дати завершення: " + ', '.join(map(str, completion_dates)) + "\n\n"
            output += "Відмінності: " + ', '.join(map(str, differences)) + "\n\n"
            output += "Різниці між f_optimum і f_solution:\n"

            for l in range(L):
                optimum_value = self.find_optimal_solution_task1(production_data, l)[2]
                f_solution = sum(production_data[7][l][i] * y_solution[i] for i in range(num_production_factors)) - sum(
                    production_data[8][i] * z_solution[i] for i in range(num_assigned_products))
                difference = optimum_value - f_solution
                output += f"{l = }, омега_l = {production_data[13][l]:,.2f}, оптимальне значення = {optimum_value:,.2f}, значення рішення = {f_solution:,.2f}, різниця = {difference:,.2f}\n"

            self.task1_output_text.delete(1.0, tk.END)
            self.task1_output_text.insert(tk.END, output)

        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def find_optimal_solution_task1(self, production_data, l):
        num_aggregated_products, num_production_factors, num_assigned_products, _, production_matrix, y_assigned, b, c, f, priorities, directive_terms, t_0, alpha, _ = production_data

        lp_solver = pywraplp.Solver.CreateSolver("GLOP")

        # Визначення змінних
        y = [lp_solver.NumVar(0, lp_solver.infinity(), f"y_{i}") for i in range(num_production_factors)]
        z = [lp_solver.NumVar(0, lp_solver.infinity(), f"z_{i}") for i in range(num_assigned_products)]

        # Визначення обмежень
        for i in range(num_aggregated_products):
            lp_solver.Add(sum(production_matrix[i][j] * y[j] for j in range(num_production_factors)) <= b[i])
        for i in range(num_assigned_products):
            lp_solver.Add(z[i] >= 0)
            lp_solver.Add((t_0[i] + alpha[i] * y[i]) - z[i] <= directive_terms[i])
        for i in range(num_assigned_products):
            lp_solver.Add(y[i] >= y_assigned[i])

        # Визначення цільової функції
        objective = lp_solver.Objective()
        for i in range(num_production_factors):
            objective.SetCoefficient(y[i], c[l][i] * priorities[i])
        for i in range(num_assigned_products):
            objective.SetCoefficient(z[i], -f[i])
        objective.SetMaximization()

        lp_solver.Solve()
        return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
            num_assigned_products)], objective.Value()

    def solve_production_problem_task1(self, production_data):
        num_aggregated_products, num_production_factors, num_assigned_products, L, production_matrix, y_assigned, b, c, f, priorities, directive_terms, t_0, alpha, omega = production_data

        lp_solver = pywraplp.Solver.CreateSolver("GLOP")

        # Визначення змінних
        y = [lp_solver.NumVar(0, lp_solver.infinity(), f"y_{i}") for i in range(num_production_factors)]
        z = [lp_solver.NumVar(0, lp_solver.infinity(), f"z_{i}") for i in range(num_assigned_products)]

        # Визначення обмежень
        for i in range(num_aggregated_products):
            lp_solver.Add(sum(production_matrix[i][j] * y[j] for j in range(num_production_factors)) <= b[i])
        for i in range(num_assigned_products):
            lp_solver.Add(z[i] >= 0)
            lp_solver.Add((t_0[i] + alpha[i] * y[i]) - z[i] <= directive_terms[i])
        for i in range(num_assigned_products):
            lp_solver.Add(y[i] >= y_assigned[i])

        # Визначення цільової функції
        objective = lp_solver.Objective()
        for i, omega_value in enumerate(omega):
            for l in range(num_production_factors):
                objective.SetCoefficient(y[l], c[i][l] * priorities[l] * omega_value)
            for l in range(num_assigned_products):
                objective.SetCoefficient(z[l], -f[l] * omega_value)
        objective.SetMaximization()

        lp_solver.Solve()
        return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
            num_assigned_products)], objective.Value()

    def load_task1_from_file(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            if not file_path:
                return

            with open(file_path, 'r') as file:
                lines = file.readlines()

            self.task1_entries["Кількість агрегованих продуктів:"].insert(0, lines[0].strip())
            self.task1_entries["Кількість факторів виробництва:"].insert(0, lines[1].strip())
            self.task1_entries["Кількість призначених продуктів:"].insert(0, lines[2].strip())
            self.task1_entries["Значення L:"].insert(0, lines[3].strip())
            self.task1_entries["Матриця виробництва (построчно, через кому):"].insert(0, lines[4].strip())
            self.task1_entries["Призначені значення Y (через кому):"].insert(0, lines[5].strip())
            self.task1_entries["Значення B (через кому):"].insert(0, lines[6].strip())
            self.task1_entries["Матриця значень C (через кому):"].insert(0, lines[7].strip())
            self.task1_entries["Значення F (через кому):"].insert(0, lines[8].strip())
            self.task1_entries["Значення пріоритетів (через кому):"].insert(0, lines[9].strip())
            self.task1_entries["Значення директивних термінів (через кому):"].insert(0, lines[10].strip())
            self.task1_entries["Значення T_0 (через кому):"].insert(0, lines[11].strip())
            self.task1_entries["Значення альфа (через кому):"].insert(0, lines[12].strip())
            self.task1_entries["Значення омега (через кому):"].insert(0, lines[13].strip())

        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def create_task2_widgets(self):
        self.task2_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.task2_tab, text="Задача 2")

        labels = [
            "Кількість агрегованих продуктів:",
            "Кількість факторів виробництва:",
            "Кількість призначених продуктів:",
            "Матриця виробництва (построчно, через кому):",
            "Призначені значення Y (через кому):",
            "Значення B (через кому):",
            "Значення C (через кому):",
            "Значення F (через кому):",
            "Значення пріоритетів (через кому):",
            "Значення директивних термінів (через кому):",
            "Значення T_0 (через кому):",
            "Значення альфа (через кому):"
        ]

        self.task2_entries = {}
        for i, label in enumerate(labels):
            ttk.Label(self.task2_tab, text=label).grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
            self.task2_entries[label] = ttk.Entry(self.task2_tab, width=50)
            self.task2_entries[label].grid(row=i, column=1, padx=10, pady=5)

        ttk.Button(self.task2_tab, text="Розв'язати задачу 2", command=self.solve_task2).grid(row=len(labels), column=0, columnspan=2, pady=10)
        ttk.Button(self.task2_tab, text="Завантажити з файлу", command=self.load_task2_from_file).grid(row=len(labels) + 1, column=0, columnspan=2, pady=10)
        self.task2_output_text = tk.Text(self.task2_tab, wrap=tk.WORD, width=100, height=20)
        self.task2_output_text.grid(row=len(labels) + 2, column=0, columnspan=2, pady=10)

    def solve_task2(self):
        try:
            num_aggregated_products = int(self.task2_entries["Кількість агрегованих продуктів:"].get())
            num_production_factors = int(self.task2_entries["Кількість факторів виробництва:"].get())
            num_assigned_products = int(self.task2_entries["Кількість призначених продуктів:"].get())

            production_matrix = np.array(
                list(map(float, self.task2_entries["Матриця виробництва (построчно, через кому):"].get().split(','))),
                dtype=float).reshape((num_aggregated_products, num_production_factors))
            y_assigned = list(map(float, self.task2_entries["Призначені значення Y (через кому):"].get().split(',')))
            b = list(map(float, self.task2_entries["Значення B (через кому):"].get().split(',')))
            c = list(map(float, self.task2_entries["Значення C (через кому):"].get().split(',')))
            f = list(map(float, self.task2_entries["Значення F (через кому):"].get().split(',')))
            priorities = list(map(float, self.task2_entries["Значення пріоритетів (через кому):"].get().split(',')))
            directive_terms = list(
                map(float, self.task2_entries["Значення директивних термінів (через кому):"].get().split(',')))
            t_0 = list(map(float, self.task2_entries["Значення T_0 (через кому):"].get().split(',')))
            alpha = list(map(float, self.task2_entries["Значення альфа (через кому):"].get().split(',')))

            production_data = (
                num_aggregated_products, num_production_factors, num_assigned_products, production_matrix, y_assigned, b,
                c, f, priorities, directive_terms, t_0, alpha)

            y_solution, z_solution, objective_value = self.solve_production_problem_task2(production_data)
            t_0 = production_data[10]
            alpha = production_data[11]
            policy_deadlines = [t_0[i] + alpha[i] * y_solution[i] for i in range(num_assigned_products)]
            completion_dates = [z_solution[i] for i in range(num_assigned_products)]
            differences = [policy_deadlines[i] - completion_dates[i] for i in range(num_assigned_products)]

            output = f"Цільове значення: {objective_value}\n\n"
            output += "Рішення Y: " + ', '.join(map(str, y_solution)) + "\n\n"
            output += "Рішення Z: " + ', '.join(map(str, z_solution)) + "\n\n"
            output += "Директивні терміни: " + ', '.join(map(str, policy_deadlines)) + "\n\n"
            output += "Дати завершення: " + ', '.join(map(str, completion_dates)) + "\n\n"
            output += "Відмінності: " + ', '.join(map(str, differences)) + "\n\n"

            self.task2_output_text.delete(1.0, tk.END)
            self.task2_output_text.insert(tk.END, output)

        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def solve_production_problem_task2(self, production_data):
        num_aggregated_products, num_production_factors, num_assigned_products, production_matrix, y_assigned, b, c, f, priorities, directive_terms, t_0, alpha = production_data

        lp_solver = pywraplp.Solver.CreateSolver("GLOP")

        # Визначення змінних
        y = [lp_solver.NumVar(0, lp_solver.infinity(), f"y_{i}") for i in range(num_production_factors)]
        z = [lp_solver.NumVar(0, lp_solver.infinity(), f"z_{i}") for i in range(num_assigned_products)]

        # Визначення обмежень
        for i in range(num_aggregated_products):
            lp_solver.Add(sum(production_matrix[i][j] * y[j] for j in range(num_production_factors)) <= b[i])
        for i in range(num_assigned_products):
            lp_solver.Add(z[i] >= 0)
            lp_solver.Add((t_0[i] + alpha[i] * y[i]) - z[i] <= directive_terms[i])
        for i in range(num_assigned_products):
            lp_solver.Add(y[i] >= y_assigned[i])

        # Визначення цільової функції
        objective = lp_solver.Objective()
        for l in range(num_production_factors):
            objective.SetCoefficient(y[l], c[l] * priorities[l])
        for i in range(num_assigned_products):
            objective.SetCoefficient(z[i], -f[i])
        objective.SetMaximization()

        lp_solver.Solve()
        return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
            num_assigned_products)], objective.Value()

    def load_task2_from_file(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            if not file_path:
                return

            with open(file_path, 'r') as file:
                lines = file.readlines()

            self.task2_entries["Кількість агрегованих продуктів:"].insert(0, lines[0].strip())
            self.task2_entries["Кількість факторів виробництва:"].insert(0, lines[1].strip())
            self.task2_entries["Кількість призначених продуктів:"].insert(0, lines[2].strip())
            self.task2_entries["Матриця виробництва (построчно, через кому):"].insert(0, lines[3].strip())
            self.task2_entries["Призначені значення Y (через кому):"].insert(0, lines[4].strip())
            self.task2_entries["Значення B (через кому):"].insert(0, lines[5].strip())
            self.task2_entries["Значення C (через кому):"].insert(0, lines[6].strip())
            self.task2_entries["Значення F (через кому):"].insert(0, lines[7].strip())
            self.task2_entries["Значення пріоритетів (через кому):"].insert(0, lines[8].strip())
            self.task2_entries["Значення директивних термінів (через кому):"].insert(0, lines[9].strip())
            self.task2_entries["Значення T_0 (через кому):"].insert(0, lines[10].strip())
            self.task2_entries["Значення альфа (через кому):"].insert(0, lines[11].strip())

        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def create_task3_widgets(self):
        self.task3_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.task3_tab, text="Задача 3")

        labels = [
            "Кількість агрегованих продуктів:",
            "Кількість факторів виробництва:",
            "Кількість призначених продуктів:",
            "Значення L:",
            "Значення M_L:",
            "Матриця виробництва (построчно, через кому):",
            "Призначені значення Y (через кому):",
            "Значення B (через кому):",
            "Матриця значень C_L (через кому):",
            "Значення F (через кому):",
            "Значення пріоритетів (через кому):",
            "Значення директивних термінів (через кому):",
            "Значення T_0 (через кому):",
            "Значення альфа (через кому):",
            "Значення омега (через кому):",
            "Значення P_L (через кому):"
        ]

        self.task3_entries = {}
        for i, label in enumerate(labels):
            ttk.Label(self.task3_tab, text=label).grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
            self.task3_entries[label] = ttk.Entry(self.task3_tab, width=50)
            self.task3_entries[label].grid(row=i, column=1, padx=10, pady=5)

        ttk.Button(self.task3_tab, text="Розв'язати задачу 3", command=self.solve_task3).grid(row=len(labels), column=0, columnspan=2, pady=10)
        ttk.Button(self.task3_tab, text="Завантажити з файлу", command=self.load_task3_from_file).grid(row=len(labels) + 1, column=0, columnspan=2, pady=10)
        self.task3_output_text = tk.Text(self.task3_tab, wrap=tk.WORD, width=100, height=20)
        self.task3_output_text.grid(row=len(labels) + 2, column=0, columnspan=2, pady=10)

    def solve_task3(self):
        try:
            num_aggregated_products = int(self.task3_entries["Кількість агрегованих продуктів:"].get())
            num_production_factors = int(self.task3_entries["Кількість факторів виробництва:"].get())
            num_assigned_products = int(self.task3_entries["Кількість призначених продуктів:"].get())
            L = int(self.task3_entries["Значення L:"].get())
            M_L = int(self.task3_entries["Значення M_L:"].get())

            production_matrix = np.array(
                list(map(float, self.task3_entries["Матриця виробництва (построчно, через кому):"].get().split(','))),
                dtype=float).reshape((num_aggregated_products, num_production_factors))
            y_assigned = list(map(float, self.task3_entries["Призначені значення Y (через кому):"].get().split(',')))
            b = list(map(float, self.task3_entries["Значення B (через кому):"].get().split(',')))

            c_l_values = list(map(float, self.task3_entries["Матриця значень C_L (через кому):"].get().split(',')))
            C_L = np.array(c_l_values, dtype=float).reshape((M_L, L, num_production_factors))

            f = list(map(float, self.task3_entries["Значення F (через кому):"].get().split(',')))
            priorities = list(map(float, self.task3_entries["Значення пріоритетів (через кому):"].get().split(',')))
            directive_terms = list(
                map(float, self.task3_entries["Значення директивних термінів (через кому):"].get().split(',')))
            t_0 = list(map(float, self.task3_entries["Значення T_0 (через кому):"].get().split(',')))
            alpha = list(map(float, self.task3_entries["Значення альфа (через кому):"].get().split(',')))
            omega = list(map(float, self.task3_entries["Значення омега (через кому):"].get().split(',')))
            P_L = list(map(float, self.task3_entries["Значення P_L (через кому):"].get().split(',')))

            production_data = (
                num_aggregated_products, num_production_factors, num_assigned_products, L, M_L, production_matrix,
                y_assigned, b, C_L, f, priorities, directive_terms, t_0, alpha, omega, P_L)

            F_L_M_optimums = []
            for m in range(M_L):
                inner_optimums = []
                for l in range(L):
                    temp_test_production_data = (
                        production_matrix, y_assigned, b, C_L[m][l], f, priorities, directive_terms, t_0, alpha)
                    _, _, objective_value = self.find_temp_optimal_solution_task3(temp_test_production_data,
                                                                                 num_production_factors,
                                                                                 num_assigned_products,
                                                                                 num_aggregated_products)
                    inner_optimums.append(objective_value)
                F_L_M_optimums.append(inner_optimums)

            y_solution, z_solution, objective_value = self.solve_production_problem_task3(production_data)
            policy_deadlines = [t_0[i] + alpha[i] * y_solution[i] for i in range(num_assigned_products)]
            completion_dates = [z_solution[i] for i in range(num_assigned_products)]
            differences = [policy_deadlines[i] - completion_dates[i] for i in range(num_assigned_products)]

            output = f"Цільове значення: {objective_value}\n\n"
            output += "Рішення Y: " + ', '.join(map(str, y_solution)) + "\n\n"
            output += "Рішення Z: " + ', '.join(map(str, z_solution)) + "\n\n"
            output += "Директивні терміни: " + ', '.join(map(str, policy_deadlines)) + "\n\n"
            output += "Дати завершення: " + ', '.join(map(str, completion_dates)) + "\n\n"
            output += "Відмінності: " + ', '.join(map(str, differences)) + "\n\n"
            output += "Різниці між f_optimum і f_solution:\n"

            for l in range(L):
                mean_difference = 0
                weighted_optimum_values = 0
                weighted_inner_differences = 0
                for m in range(M_L):
                    inner_difference = 0
                    c_m_l = C_L[m][l]
                    for i in range(num_assigned_products):
                        inner_difference += c_m_l[i] * y_solution[i] - f[i] * z_solution[i]
                    optimum_value = F_L_M_optimums[m][l]
                    weighted_optimum_values += P_L[m] * optimum_value
                    weighted_inner_differences += P_L[m] * inner_difference
                mean_difference = weighted_optimum_values - weighted_inner_differences
                output += f"{l = }, омега_l = {omega[l]:,.2f}, зважене оптимальне значення = {weighted_optimum_values:,.2f}, зважені внутрішні відмінності = {weighted_inner_differences:,.2f}, середнє відхилення = {mean_difference:,.2f}\n"

            self.task3_output_text.delete(1.0, tk.END)
            self.task3_output_text.insert(tk.END, output)

        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def find_temp_optimal_solution_task3(self, temp_production_data, num_production_factors, num_assigned_products,
                                         num_aggregated_products):
        production_matrix, y_assigned, b, c_l_m, f, priorities, directive_terms, t_0, alpha = temp_production_data

        lp_solver = pywraplp.Solver.CreateSolver("GLOP")

        # Визначення змінних
        y = [lp_solver.NumVar(0, lp_solver.infinity(), f"y_{i}") for i in range(num_production_factors)]
        z = [lp_solver.NumVar(0, lp_solver.infinity(), f"z_{i}") for i in range(num_assigned_products)]

        # Визначення обмежень
        for i in range(num_aggregated_products):
            lp_solver.Add(sum(production_matrix[i][j] * y[j] for j in range(num_production_factors)) <= b[i])
        for i in range(num_assigned_products):
            lp_solver.Add(z[i] >= 0)
            lp_solver.Add((t_0[i] + alpha[i] * y[i]) - z[i] <= directive_terms[i])
        for i in range(num_assigned_products):
            lp_solver.Add(y[i] >= y_assigned[i])

        # Визначення цільової функції
        objective = lp_solver.Objective()
        for i in range(num_production_factors):
            objective.SetCoefficient(y[i], c_l_m[i] * priorities[i])
        for i in range(num_assigned_products):
            objective.SetCoefficient(z[i], -f[i])
        objective.SetMaximization()

        lp_solver.Solve()
        return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
            num_assigned_products)], objective.Value()

    def solve_production_problem_task3(self, production_data):
        num_aggregated_products, num_production_factors, num_assigned_products, L, M_L, production_matrix, y_assigned, b, C_L, f, priorities, directive_terms, t_0, alpha, omega, P_L = production_data

        lp_solver = pywraplp.Solver.CreateSolver("GLOP")

        # Визначення змінних
        y = [lp_solver.NumVar(0, lp_solver.infinity(), f"y_{i}") for i in range(num_production_factors)]
        z = [lp_solver.NumVar(0, lp_solver.infinity(), f"z_{i}") for i in range(num_assigned_products)]

        # Визначення обмежень
        for i in range(num_aggregated_products):
            lp_solver.Add(sum(production_matrix[i][j] * y[j] for j in range(num_production_factors)) <= b[i])
        for i in range(num_assigned_products):
            lp_solver.Add(z[i] >= 0)
            lp_solver.Add((t_0[i] + alpha[i] * y[i]) - z[i] <= directive_terms[i])
        for i in range(num_assigned_products):
            lp_solver.Add(y[i] >= y_assigned[i])

        # Визначення цільової функції
        objective = lp_solver.Objective()
        for l in range(L):
            for m in range(M_L):
                for i in range(num_production_factors):
                    objective.SetCoefficient(y[i], C_L[m][l][i] * priorities[i] * omega[l] * P_L[m])
                for i in range(num_assigned_products):
                    objective.SetCoefficient(z[i], -f[i] * P_L[m])
        objective.SetMaximization()

        lp_solver.Solve()
        return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
            num_assigned_products)], objective.Value()

    def load_task3_from_file(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            if not file_path:
                return

            with open(file_path, 'r') as file:
                lines = file.readlines()

            if len(lines) != 16:
                raise ValueError("Файл введення повинен містити точно 16 рядків.")

            for key in self.task3_entries:
                self.task3_entries[key].delete(0, tk.END)

            self.task3_entries["Кількість агрегованих продуктів:"].insert(0, lines[0].strip())
            self.task3_entries["Кількість факторів виробництва:"].insert(0, lines[1].strip())
            self.task3_entries["Кількість призначених продуктів:"].insert(0, lines[2].strip())
            self.task3_entries["Значення L:"].insert(0, lines[3].strip())
            self.task3_entries["Значення M_L:"].insert(0, lines[4].strip())
            self.task3_entries["Матриця виробництва (построчно, через кому):"].insert(0, lines[5].strip())
            self.task3_entries["Призначені значення Y (через кому):"].insert(0, lines[6].strip())
            self.task3_entries["Значення B (через кому):"].insert(0, lines[7].strip())
            self.task3_entries["Матриця значень C_L (через кому):"].insert(0, lines[8].strip())
            self.task3_entries["Значення F (через кому):"].insert(0, lines[9].strip())
            self.task3_entries["Значення пріоритетів (через кому):"].insert(0, lines[10].strip())
            self.task3_entries["Значення директивних термінів (через кому):"].insert(0, lines[11].strip())
            self.task3_entries["Значення T_0 (через кому):"].insert(0, lines[12].strip())
            self.task3_entries["Значення альфа (через кому):"].insert(0, lines[13].strip())
            self.task3_entries["Значення омега (через кому):"].insert(0, lines[14].strip())
            self.task3_entries["Значення P_L (через кому):"].insert(0, lines[15].strip())

        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def create_task4_widgets(self):
        self.task4_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.task4_tab, text="Задача 4")

        labels = [
            "Кількість агрегованих продуктів:",
            "Кількість факторів виробництва:",
            "Кількість призначених продуктів:",
            "Значення M:",
            "Матриця виробництва (построчно, через кому):",
            "Призначені значення Y (через кому):",
            "Значення B (через кому):",
            "Матриця значень C (через кому):",
            "Значення P_m (через кому):",
            "Значення F (через кому):",
            "Значення пріоритетів (через кому):",
            "Значення директивних термінів (через кому):",
            "Значення T_0 (через кому):",
            "Значення альфа (через кому):",
            "Значення омега (через кому):"
        ]

        self.task4_entries = {}
        for i, label in enumerate(labels):
            ttk.Label(self.task4_tab, text=label).grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
            self.task4_entries[label] = ttk.Entry(self.task4_tab, width=50)
            self.task4_entries[label].grid(row=i, column=1, padx=10, pady=5)

        ttk.Button(self.task4_tab, text="Розв'язати задачу 4", command=self.solve_task4).grid(row=len(labels), column=0, columnspan=2, pady=10)
        ttk.Button(self.task4_tab, text="Завантажити з файлу", command=self.load_task4_from_file).grid(row=len(labels) + 1, column=0, columnspan=2, pady=10)
        self.task4_output_text = tk.Text(self.task4_tab, wrap=tk.WORD, width=100, height=20)
        self.task4_output_text.grid(row=len(labels) + 2, column=0, columnspan=2, pady=10)

    def solve_task4(self):
        try:
            num_aggregated_products = int(self.task4_entries["Кількість агрегованих продуктів:"].get())
            num_production_factors = int(self.task4_entries["Кількість факторів виробництва:"].get())
            num_assigned_products = int(self.task4_entries["Кількість призначених продуктів:"].get())
            M = int(self.task4_entries["Значення M:"].get())

            production_matrix = np.array(
                list(map(float, self.task4_entries["Матриця виробництва (построчно, через кому):"].get().split(','))),
                dtype=float).reshape((num_aggregated_products, num_production_factors))
            y_assigned = list(map(float, self.task4_entries["Призначені значення Y (через кому):"].get().split(',')))
            b = list(map(float, self.task4_entries["Значення B (через кому):"].get().split(',')))

            c_values = list(map(float, self.task4_entries["Матриця значень C (через кому):"].get().split(',')))
            c = np.array(c_values, dtype=float).reshape((M, num_production_factors))

            P_m = list(map(float, self.task4_entries["Значення P_m (через кому):"].get().split(',')))
            f = list(map(float, self.task4_entries["Значення F (через кому):"].get().split(',')))
            priorities = list(map(float, self.task4_entries["Значення пріоритетів (через кому):"].get().split(',')))
            directive_terms = list(
                map(float, self.task4_entries["Значення директивних термінів (через кому):"].get().split(',')))
            t_0 = list(map(float, self.task4_entries["Значення T_0 (через кому):"].get().split(',')))
            alpha = list(map(float, self.task4_entries["Значення альфа (через кому):"].get().split(',')))
            omega = list(map(float, self.task4_entries["Значення омега (через кому):"].get().split(',')))

            production_data = (
                num_aggregated_products, num_production_factors, num_assigned_products, M, production_matrix,
                y_assigned, b, c, P_m, f, priorities, directive_terms, t_0, alpha, omega)

            F_optimums = [self.find_temp_optimal_solution_task4(production_data, m)[2] for m in range(M)]
            y_solution, z_solution, objective_value = self.solve_production_problem_task4(production_data)
            t_0 = production_data[12]
            alpha = production_data[13]
            policy_deadlines = [t_0[i] + alpha[i] * y_solution[i] for i in range(num_assigned_products)]
            completion_dates = [z_solution[i] for i in range(num_assigned_products)]
            differences = [policy_deadlines[i] - completion_dates[i] for i in range(num_assigned_products)]

            output = f"Цільове значення: {objective_value}\n\n"
            output += "Рішення Y: " + ', '.join(map(str, y_solution)) + "\n\n"
            output += "Рішення Z: " + ', '.join(map(str, z_solution)) + "\n\n"
            output += "Директивні терміни: " + ', '.join(map(str, policy_deadlines)) + "\n\n"
            output += "Дати завершення: " + ', '.join(map(str, completion_dates)) + "\n\n"
            output += "Відмінності: " + ', '.join(map(str, differences)) + "\n\n"
            output += "Різниці між f_optimum і f_solution:\n"

            for m in range(M):
                c_l_m = production_data[7][m]
                f = production_data[9]
                f_solution = sum([c_l_m[i] * y_solution[i] for i in range(num_production_factors)]) - sum(
                    [f[i] * z_solution[i] for i in range(num_assigned_products)])
                difference = F_optimums[m] - f_solution
                output += f"{m = }, F_optimums[m] = {F_optimums[m]:,.2f}, f_solution = {f_solution:,.2f}, різниця = {difference:,.2f}\n"

            self.task4_output_text.delete(1.0, tk.END)
            self.task4_output_text.insert(tk.END, output)

        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def find_temp_optimal_solution_task4(self, production_data, m):
        num_aggregated_products, num_production_factors, num_assigned_products, _, production_matrix, y_assigned, b, c, _, f, priorities, directive_terms, t_0, alpha, _ = production_data

        lp_solver = pywraplp.Solver.CreateSolver("GLOP")

        # Визначення змінних
        y = [lp_solver.NumVar(0, lp_solver.infinity(), f"y_{i}") for i in range(num_production_factors)]
        z = [lp_solver.NumVar(0, lp_solver.infinity(), f"z_{i}") for i in range(num_assigned_products)]

        # Визначення обмежень
        for i in range(num_aggregated_products):
            lp_solver.Add(sum(production_matrix[i][j] * y[j] for j in range(num_production_factors)) <= b[i])
        for i in range(num_assigned_products):
            lp_solver.Add(z[i] >= 0)
            lp_solver.Add((t_0[i] + alpha[i] * y[i]) - z[i] <= directive_terms[i])
        for i in range(num_assigned_products):
            lp_solver.Add(y[i] >= y_assigned[i])

        # Визначення цільової функції
        objective = lp_solver.Objective()
        for l in range(num_production_factors):
            objective.SetCoefficient(y[l], c[m][l] * priorities[l])
        for i in range(num_assigned_products):
            objective.SetCoefficient(z[i], -f[i])
        objective.SetMaximization()

        lp_solver.Solve()
        return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
            num_assigned_products)], objective.Value()

    def solve_production_problem_task4(self, production_data):
        num_aggregated_products, num_production_factors, num_assigned_products, M, production_matrix, y_assigned, b, c, P_m, f, priorities, directive_terms, t_0, alpha, omega = production_data

        lp_solver = pywraplp.Solver.CreateSolver("GLOP")

        # Визначення змінних
        y = [lp_solver.NumVar(0, lp_solver.infinity(), f"y_{i}") for i in range(num_production_factors)]
        z = [lp_solver.NumVar(0, lp_solver.infinity(), f"z_{i}") for i in range(num_assigned_products)]

        # Визначення обмежень
        for i in range(num_aggregated_products):
            lp_solver.Add(sum(production_matrix[i][j] * y[j] for j in range(num_production_factors)) <= b[i])
        for i in range(num_assigned_products):
            lp_solver.Add(z[i] >= 0)
            lp_solver.Add((t_0[i] + alpha[i] * y[i]) - z[i] <= directive_terms[i])
        for i in range(num_assigned_products):
            lp_solver.Add(y[i] >= y_assigned[i])

        # Визначення цільової функції
        objective = lp_solver.Objective()
        for i, p_m in enumerate(P_m):
            for l in range(num_production_factors):
                objective.SetCoefficient(y[l], c[i][l] * priorities[l] * omega[l] * p_m)
            for l in range(num_assigned_products):
                objective.SetCoefficient(z[l], -f[l] * p_m)
        objective.SetMaximization()

        lp_solver.Solve()
        return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
            num_assigned_products)], objective.Value()

    def load_task4_from_file(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
            if not file_path:
                return

            with open(file_path, 'r') as file:
                lines = file.readlines()

            if len(lines) != 15:
                raise ValueError("Файл введення повинен містити точно 15 рядків.")

            for key in self.task4_entries:
                self.task4_entries[key].delete(0, tk.END)

            self.task4_entries["Кількість агрегованих продуктів:"].insert(0, lines[0].strip())
            self.task4_entries["Кількість факторів виробництва:"].insert(0, lines[1].strip())
            self.task4_entries["Кількість призначених продуктів:"].insert(0, lines[2].strip())
            self.task4_entries["Значення M:"].insert(0, lines[3].strip())
            self.task4_entries["Матриця виробництва (построчно, через кому):"].insert(0, lines[4].strip())
            self.task4_entries["Призначені значення Y (через кому):"].insert(0, lines[5].strip())
            self.task4_entries["Значення B (через кому):"].insert(0, lines[6].strip())
            self.task4_entries["Матриця значень C (через кому):"].insert(0, lines[7].strip())
            self.task4_entries["Значення P_m (через кому):"].insert(0, lines[8].strip())
            self.task4_entries["Значення F (через кому):"].insert(0, lines[9].strip())
            self.task4_entries["Значення пріоритетів (через кому):"].insert(0, lines[10].strip())
            self.task4_entries["Значення директивних термінів (через кому):"].insert(0, lines[11].strip())
            self.task4_entries["Значення T_0 (через кому):"].insert(0, lines[12].strip())
            self.task4_entries["Значення альфа (через кому):"].insert(0, lines[13].strip())
            self.task4_entries["Значення омега (через кому):"].insert(0, lines[14].strip())

        except Exception as e:
            messagebox.showerror("Помилка", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiTaskApp(root)
    root.mainloop()
