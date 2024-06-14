import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from ortools.linear_solver import pywraplp

np.set_printoptions(linewidth=1000)


class ProductionProblemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Розв'язувач виробничих задач")

        self.tab_control = ttk.Notebook(root)

        self.example1_tab = ttk.Frame(self.tab_control)
        self.example2_tab = ttk.Frame(self.tab_control)
        self.example3_tab = ttk.Frame(self.tab_control)
        self.example4_tab = ttk.Frame(self.tab_control)

        self.tab_control.add(self.example1_tab, text="Приклад 1")
        self.tab_control.add(self.example2_tab, text="Приклад 2")
        self.tab_control.add(self.example3_tab, text="Приклад 3")
        self.tab_control.add(self.example4_tab, text="Приклад 4")

        self.tab_control.pack(expand=1, fill="both")

        self.create_task1_widgets()
        self.create_task2_widgets()
        self.create_task3_widgets()
        self.create_task4_widgets()

    def create_task1_widgets(self):
        self.entries_task1 = {}
        labels_task1 = [
            "Кількість агрегованих продуктів:",
            "Кількість факторів виробництва:",
            "Кількість призначених продуктів:",
            "Значення L:",
            "Матриця виробництва (построчно, через коми):",
            "Значення Y призначені (через коми):",
            "Значення B (через коми):",
            "Значення C матриця (через коми):",
            "Значення F (через коми):",
            "Значення пріоритетів (через коми):",
            "Значення директивних строків(через коми):",
            "Значення T_0 (через коми):",
            "Значення Alpha (через коми):",
            "Значення Omega (через коми):"
        ]

        for i, label in enumerate(labels_task1):
            ttk.Label(self.example1_tab, text=label).grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
            self.entries_task1[label] = ttk.Entry(self.example1_tab, width=50)
            self.entries_task1[label].grid(row=i, column=1, padx=10, pady=5)

        ttk.Button(self.example1_tab, text="Розв'язати", command=self.solve_production_problem_task1).grid(row=len(labels_task1), column=0, columnspan=2, pady=10)
        ttk.Button(self.example1_tab, text="Завантажити з файлу", command=self.load_from_file_task1).grid(row=len(labels_task1) + 1, column=0, columnspan=2, pady=10)
        self.output_text_task1 = tk.Text(self.example1_tab, wrap=tk.WORD, width=100, height=20)
        self.output_text_task1.grid(row=len(labels_task1) + 2, column=0, columnspan=2, pady=10)

    def create_task2_widgets(self):
        self.entries_task2 = {}
        labels_task2 = [
            "Кількість агрегованих продуктів:",
            "Кількість факторів виробництва:",
            "Кількість призначених продуктів:",
            "Значення L:",
            "Матриця виробництва (построчно, через коми):",
            "Значення Y призначені (через коми):",
            "Значення B (через коми):",
            "Значення C матриця (через коми):",
            "Значення F (через коми):",
            "Значення пріоритетів (через коми):",
            "Значення директивних строків(через коми):",
            "Значення T_0 (через коми):",
            "Значення Alpha (через коми):"
        ]

        for i, label in enumerate(labels_task2):
            ttk.Label(self.example2_tab, text=label).grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
            self.entries_task2[label] = ttk.Entry(self.example2_tab, width=50)
            self.entries_task2[label].grid(row=i, column=1, padx=10, pady=5)

        ttk.Button(self.example2_tab, text="Розв'язати", command=self.solve_production_problem_task2).grid(row=len(labels_task2), column=0, columnspan=2, pady=10)
        ttk.Button(self.example2_tab, text="Завантажити з файлу", command=self.load_from_file_task2).grid(row=len(labels_task2) + 1, column=0, columnspan=2, pady=10)
        self.output_text_task2 = tk.Text(self.example2_tab, wrap=tk.WORD, width=100, height=20)
        self.output_text_task2.grid(row=len(labels_task2) + 2, column=0, columnspan=2, pady=10)

    def create_task3_widgets(self):
        self.entries_task3 = {}
        labels_task3 = [
            "Кількість агрегованих продуктів:",
            "Кількість факторів виробництва:",
            "Кількість призначених продуктів:",
            "Значення L:",
            "Матриця виробництва (построчно, через коми):",
            "Значення Y призначені (через коми):",
            "Значення B (через коми):",
            "Значення C матриця (через коми):",
            "Значення F (через коми):",
            "Значення пріоритетів (через коми):",
            "Значення директивних строків(через коми):",
            "Значення T_0 (через коми):",
            "Значення Alpha (через коми):",
            "Значення Omega (через коми):"
        ]

        for i, label in enumerate(labels_task3):
            ttk.Label(self.example3_tab, text=label).grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
            self.entries_task3[label] = ttk.Entry(self.example3_tab, width=50)
            self.entries_task3[label].grid(row=i, column=1, padx=10, pady=5)

        ttk.Button(self.example3_tab, text="Розв'язати", command=self.solve_production_problem_task3).grid(row=len(labels_task3), column=0, columnspan=2, pady=10)
        ttk.Button(self.example3_tab, text="Завантажити з файлу", command=self.load_from_file_task3).grid(row=len(labels_task3) + 1, column=0, columnspan=2, pady=10)
        self.output_text_task3 = tk.Text(self.example3_tab, wrap=tk.WORD, width=100, height=20)
        self.output_text_task3.grid(row=len(labels_task3) + 2, column=0, columnspan=2, pady=10)

    def create_task4_widgets(self):
        self.entries_task4 = {}
        labels_task4 = [
            "Кількість агрегованих продуктів:",
            "Кількість факторів виробництва:",
            "Кількість призначених продуктів:",
            "Значення L:",
            "Значення M_L:",
            "Матриця виробництва (построчно, через коми):",
            "Значення Y призначені (через коми):",
            "Значення B (через коми):",
            "Значення C_L (через коми):",
            "Значення F (через коми):",
            "Значення пріоритетів (через коми):",
            "Значення директивних строків(через коми):",
            "Значення T_0 (через коми):",
            "Значення Alpha (через коми):",
            "Значення Omega (через коми):",
            "Значення P_L (через коми):"
        ]

        for i, label in enumerate(labels_task4):
            ttk.Label(self.example4_tab, text=label).grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)
            self.entries_task4[label] = ttk.Entry(self.example4_tab, width=50)
            self.entries_task4[label].grid(row=i, column=1, padx=10, pady=5)

        ttk.Button(self.example4_tab, text="Розв'язати", command=self.solve_production_problem_task4).grid(row=len(labels_task4), column=0, columnspan=2, pady=10)
        ttk.Button(self.example4_tab, text="Завантажити з файлу", command=self.load_from_file_task4).grid(row=len(labels_task4) + 1, column=0, columnspan=2, pady=10)
        self.output_text_task4 = tk.Text(self.example4_tab, wrap=tk.WORD, width=100, height=20)
        self.output_text_task4.grid(row=len(labels_task4) + 2, column=0, columnspan=2, pady=10)

    def get_input(self, label, task):
        if task == 1:
            return self.entries_task1[label].get()
        elif task == 2:
            return self.entries_task2[label].get()
        elif task == 3:
            return self.entries_task3[label].get()
        elif task == 4:
            return self.entries_task4[label].get()

    def solve_production_problem_task1(self):
        try:
            num_aggregated_products = int(self.get_input("Кількість агрегованих продуктів:", 1))
            num_production_factors = int(self.get_input("Кількість факторів виробництва:", 1))
            num_assigned_products = int(self.get_input("Кількість призначених продуктів:", 1))
            L = int(self.get_input("Значення L:", 1))

            production_matrix = np.array(
                list(map(float, self.get_input("Матриця виробництва (построчно, через коми):", 1).split(','))),
                dtype=float).reshape((num_aggregated_products, num_production_factors))
            y_assigned = list(map(float, self.get_input("Значення Y призначені (через коми):", 1).split(',')))
            b = list(map(float, self.get_input("Значення B (через коми):", 1).split(',')))
            c = np.array(list(map(float, self.get_input("Значення C матриця (через коми):", 1).split(','))),
                         dtype=float).reshape((L, num_production_factors))
            f = list(map(float, self.get_input("Значення F (через коми):", 1).split(',')))
            priorities = list(map(float, self.get_input("Значення пріоритетів (через коми):", 1).split(',')))
            directive_terms = list(
                map(float, self.get_input("Значення директивних строків(через коми):", 1).split(',')))
            t_0 = list(map(float, self.get_input("Значення T_0 (через коми):", 1).split(',')))
            alpha = list(map(float, self.get_input("Значення Alpha (через коми):", 1).split(',')))
            omega = list(map(float, self.get_input("Значення Omega (через коми):", 1).split(',')))

            production_data = (
            num_aggregated_products, num_production_factors, num_assigned_products, L, production_matrix, y_assigned, b,
            c, f, priorities, directive_terms, t_0, alpha, omega)

            y_solution, z_solution, objective_value = solve_production_problem_task1(production_data)
            t_0 = production_data[11]
            alpha = production_data[12]
            policy_deadlines = [t_0[i] + alpha[i] * y_solution[i] for i in range(num_assigned_products)]
            completion_dates = [z_solution[i] for i in range(num_assigned_products)]
            differences = [policy_deadlines[i] - completion_dates[i] for i in range(num_assigned_products)]

            output = f"Цільова функція: {objective_value}\n\n"
            output += "Розв'язок Y: " + ', '.join(map(str, y_solution)) + "\n\n"
            output += "Розв'язок Z: " + ', '.join(map(str, z_solution)) + "\n\n"
            output += "Директивні строки: " + ', '.join(map(str, policy_deadlines)) + "\n\n"
            output += "Дати завершення: " + ', '.join(map(str, completion_dates)) + "\n\n"
            output += "Різниці: " + ', '.join(map(str, differences)) + "\n\n"
            output += "Різниці між f_оптимум і f_розв'язок:\n"

            for l in range(L):
                optimum_value = find_optimal_solution_task1(production_data, l)[2]
                f_solution = sum(production_data[7][l][i] * y_solution[i] for i in range(num_production_factors)) - sum(
                    production_data[8][i] * z_solution[i] for i in range(num_assigned_products))
                difference = optimum_value - f_solution
                output += f"{l = }, omega_l = {production_data[13][l]:,.2f}, optimum_value = {optimum_value:,.2f}, f_solution = {f_solution:,.2f}, difference = {difference:,.2f}\n"

            self.output_text_task1.delete(1.0, tk.END)
            self.output_text_task1.insert(tk.END, output)

        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def solve_production_problem_task2(self):
        try:
            num_aggregated_products = int(self.get_input("Кількість агрегованих продуктів:", 2))
            num_production_factors = int(self.get_input("Кількість факторів виробництва:", 2))
            num_assigned_products = int(self.get_input("Кількість призначених продуктів:", 2))
            L = int(self.get_input("Значення L:", 2))

            production_matrix = np.array(
                list(map(float, self.get_input("Матриця виробництва (построчно, через коми):", 2).split(','))),
                dtype=float).reshape((num_aggregated_products, num_production_factors))
            y_assigned = list(map(float, self.get_input("Значення Y призначені (через коми):", 2).split(',')))
            b = list(map(float, self.get_input("Значення B (через коми):", 2).split(',')))
            c = np.array(list(map(float, self.get_input("Значення C матриця (через коми):", 2).split(','))),
                         dtype=float).reshape((L, num_production_factors))
            f = list(map(float, self.get_input("Значення F (через коми):", 2).split(',')))
            priorities = list(map(float, self.get_input("Значення пріоритетів (через коми):", 2).split(',')))
            directive_terms = list(
                map(float, self.get_input("Значення директивних строків(через коми):", 2).split(',')))
            t_0 = list(map(float, self.get_input("Значення T_0 (через коми):", 2).split(',')))
            alpha = list(map(float, self.get_input("Значення Alpha (через коми):", 2).split(',')))

            production_data = (
            num_aggregated_products, num_production_factors, num_assigned_products, L, production_matrix, y_assigned, b,
            c, f, priorities, directive_terms, t_0, alpha)

            y_solution, z_solution, objective_value = solve_production_problem_task2(production_data)
            t_0 = production_data[11]
            alpha = production_data[12]
            policy_deadlines = [t_0[i] + alpha[i] * y_solution[i] for i in range(num_assigned_products)]
            completion_dates = [z_solution[i] for i in range(num_assigned_products)]
            differences = [policy_deadlines[i] - completion_dates[i] for i in range(num_assigned_products)]

            output = f"Цільова функція: {objective_value}\n\n"
            output += "Розв'язок Y: " + ', '.join(map(str, y_solution)) + "\n\n"
            output += "Розв'язок Z: " + ', '.join(map(str, z_solution)) + "\n\n"
            output += "Директивні строки: " + ', '.join(map(str, policy_deadlines)) + "\n\n"
            output += "Дати завершення: " + ', '.join(map(str, completion_dates)) + "\n\n"
            output += "Різниці: " + ', '.join(map(str, differences)) + "\n\n"
            output += "Різниці між f_оптимум і f_розв'язок:\n"

            for l in range(L):
                optimum_value = find_optimal_solution_task2(production_data, l)[2]
                f_solution = sum(production_data[7][l][i] * y_solution[i] for i in range(num_production_factors)) - sum(
                    production_data[8][i] * z_solution[i] for i in range(num_assigned_products))
                difference = optimum_value - f_solution
                output += f"{l = }, optimum_value = {optimum_value:,.2f}, f_solution = {f_solution:,.2f}, difference = {difference:,.2f}\n"

            self.output_text_task2.delete(1.0, tk.END)
            self.output_text_task2.insert(tk.END, output)

        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def solve_production_problem_task3(self):
        try:
            num_aggregated_products = int(self.get_input("Кількість агрегованих продуктів:", 3))
            num_production_factors = int(self.get_input("Кількість факторів виробництва:", 3))
            num_assigned_products = int(self.get_input("Кількість призначених продуктів:", 3))
            L = int(self.get_input("Значення L:", 3))

            production_matrix = np.array(
                list(map(float, self.get_input("Матриця виробництва (построчно, через коми):", 3).split(','))),
                dtype=float).reshape((num_aggregated_products, num_production_factors))
            y_assigned = list(map(float, self.get_input("Значення Y призначені (через коми):", 3).split(',')))
            b = list(map(float, self.get_input("Значення B (через коми):", 3).split(',')))
            c = np.array(list(map(float, self.get_input("Значення C матриця (через коми):", 3).split(','))),
                         dtype=float).reshape((L, num_production_factors))
            f = list(map(float, self.get_input("Значення F (через коми):", 3).split(',')))
            priorities = list(map(float, self.get_input("Значення пріоритетів (через коми):", 3).split(',')))
            directive_terms = list(
                map(float, self.get_input("Значення директивних строків(через коми):", 3).split(',')))
            t_0 = list(map(float, self.get_input("Значення T_0 (через коми):", 3).split(',')))
            alpha = list(map(float, self.get_input("Значення Alpha (через коми):", 3).split(',')))
            omega = list(map(float, self.get_input("Значення Omega (через коми):", 3).split(',')))

            production_data = (
            num_aggregated_products, num_production_factors, num_assigned_products, L, production_matrix, y_assigned, b,
            c, f, priorities, directive_terms, t_0, alpha, omega)

            y_solution, z_solution, objective_value = solve_production_problem_task3(production_data)
            t_0 = production_data[11]
            alpha = production_data[12]
            policy_deadlines = [t_0[i] + alpha[i] * y_solution[i] for i in range(num_assigned_products)]
            completion_dates = [z_solution[i] for i in range(num_assigned_products)]
            differences = [policy_deadlines[i] - completion_dates[i] for i in range(num_assigned_products)]

            output = f"Цільова функція: {objective_value}\n\n"
            output += "Розв'язок Y: " + ', '.join(map(str, y_solution)) + "\n\n"
            output += "Розв'язок Z: " + ', '.join(map(str, z_solution)) + "\n\n"
            output += "Директивні строки: " + ', '.join(map(str, policy_deadlines)) + "\n\n"
            output += "Дати завершення: " + ', '.join(map(str, completion_dates)) + "\n\n"
            output += "Різниці: " + ', '.join(map(str, differences)) + "\n\n"
            output += "Різниці між f_оптимум і f_розв'язок:\n"

            for l in range(L):
                optimum_value = find_optimal_solution_task3(production_data, l)[2]
                f_solution = sum(production_data[7][l][i] * y_solution[i] for i in range(num_production_factors)) - sum(
                    production_data[8][i] * z_solution[i] for i in range(num_assigned_products))
                difference = optimum_value - f_solution
                output += f"{l = }, omega_l = {production_data[13][l]:,.2f}, optimum_value = {optimum_value:,.2f}, f_solution = {f_solution:,.2f}, difference = {difference:,.2f}\n"

            self.output_text_task3.delete(1.0, tk.END)
            self.output_text_task3.insert(tk.END, output)

        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def solve_production_problem_task4(self):
        try:
            num_aggregated_products = int(self.get_input("Кількість агрегованих продуктів:", 4))
            num_production_factors = int(self.get_input("Кількість факторів виробництва:", 4))
            num_assigned_products = int(self.get_input("Кількість призначених продуктів:", 4))
            L = int(self.get_input("Значення L:", 4))
            M_L = int(self.get_input("Значення M_L:", 4))

            production_matrix = np.array(
                list(map(float, self.get_input("Матриця виробництва (построчно, через коми):", 4).split(','))),
                dtype=float).reshape((num_aggregated_products, num_production_factors))
            y_assigned = list(map(float, self.get_input("Значення Y призначені (через коми):", 4).split(',')))
            b = list(map(float, self.get_input("Значення B (через коми):", 4).split(',')))

            c_l_values = list(map(float, self.get_input("Значення C_L (через коми):", 4).split(',')))
            C_L = np.array(c_l_values, dtype=float).reshape((M_L, L, num_production_factors))

            f = list(map(float, self.get_input("Значення F (через коми):", 4).split(',')))
            priorities = list(map(float, self.get_input("Значення пріоритетів (через коми):", 4).split(',')))
            directive_terms = list(
                map(float, self.get_input("Значення директивних строків(через коми):", 4).split(',')))
            t_0 = list(map(float, self.get_input("Значення T_0 (через коми):", 4).split(',')))
            alpha = list(map(float, self.get_input("Значення Alpha (через коми):", 4).split(',')))
            omega = list(map(float, self.get_input("Значення Omega (через коми):", 4).split(',')))
            P_L = list(map(float, self.get_input("Значення P_L (через коми):", 4).split(',')))

            production_data = (
            num_aggregated_products, num_production_factors, num_assigned_products, L, M_L, production_matrix,
            y_assigned, b, C_L, f, priorities, directive_terms, t_0, alpha, omega, P_L)

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

            y_solution, z_solution, objective_value = solve_production_problem_task4(production_data)
            policy_deadlines = [t_0[i] + alpha[i] * y_solution[i] for i in range(num_assigned_products)]
            completion_dates = [z_solution[i] for i in range(num_assigned_products)]
            differences = [policy_deadlines[i] - completion_dates[i] for i in range(num_assigned_products)]

            output = f"Цільова функція: {objective_value}\n\n"
            output += "Розв'язок Y: " + ', '.join(map(str, y_solution)) + "\n\n"
            output += "Розв'язок Z: " + ', '.join(map(str, z_solution)) + "\n\n"
            output += "Директивні строки: " + ', '.join(map(str, policy_deadlines)) + "\n\n"
            output += "Дати завершення: " + ', '.join(map(str, completion_dates)) + "\n\n"
            output += "Різниці: " + ', '.join(map(str, differences)) + "\n\n"
            output += "F_L_M_оптимуми:\n\t\t"
            for l in range(L):
                output += f"{l}\t\t"
            output += "\n"
            for m in range(M_L):
                output += f"{m}\t"
                for l in range(L):
                    output += f"{F_L_M_optimums[m][l]:.2f}\t"
                output += "\n"

            output += "\nРізниці між f_оптимум і f_розв'язок:\n\t\t"
            for l in range(L):
                output += f"{l}\t\t\t"
            output += "\n"

            c = production_data[8]
            f = production_data[9]

            for m in range(M_L):
                output += f"{m}\t"
                for l in range(L):
                    inner_difference = 0
                    for i in range(num_assigned_products):
                        inner_difference += c[m][l][i] * y_solution[i] - f[i] * z_solution[i]
                    optimum_value = F_L_M_optimums[m][l]
                    difference = optimum_value - inner_difference
                    output += f"{difference:12.2f}\t"
                output += "\n"

            output += "\nСередні різниці:\n"
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
                output += f"{l = }, weighted_optimum_values = {weighted_optimum_values:,.2f}, weighted_inner_differences = {weighted_inner_differences:,.2f}, mean_difference = {mean_difference:,.2f}\n"

            self.output_text_task4.delete(1.0, tk.END)
            self.output_text_task4.insert(tk.END, output)

        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def load_from_file_task1(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Текстові файли", "*.txt")])
            if not file_path:
                return

            with open(file_path, 'r') as file:
                lines = file.readlines()

            self.entries_task1["Кількість агрегованих продуктів:"].insert(0, lines[0].strip())
            self.entries_task1["Кількість факторів виробництва:"].insert(0, lines[1].strip())
            self.entries_task1["Кількість призначених продуктів:"].insert(0, lines[2].strip())
            self.entries_task1["Значення L:"].insert(0, lines[3].strip())
            self.entries_task1["Матриця виробництва (построчно, через коми):"].insert(0, lines[4].strip())
            self.entries_task1["Значення Y призначені (через коми):"].insert(0, lines[5].strip())
            self.entries_task1["Значення B (через коми):"].insert(0, lines[6].strip())
            self.entries_task1["Значення C матриця (через коми):"].insert(0, lines[7].strip())
            self.entries_task1["Значення F (через коми):"].insert(0, lines[8].strip())
            self.entries_task1["Значення пріоритетів (через коми):"].insert(0, lines[9].strip())
            self.entries_task1["Значення директивних строків(через коми):"].insert(0, lines[10].strip())
            self.entries_task1["Значення T_0 (через коми):"].insert(0, lines[11].strip())
            self.entries_task1["Значення Alpha (через коми):"].insert(0, lines[12].strip())
            self.entries_task1["Значення Omega (через коми):"].insert(0, lines[13].strip())

        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def load_from_file_task2(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Текстові файли", "*.txt")])
            if not file_path:
                return

            with open(file_path, 'r') as file:
                lines = file.readlines()

            self.entries_task2["Кількість агрегованих продуктів:"].insert(0, lines[0].strip())
            self.entries_task2["Кількість факторів виробництва:"].insert(0, lines[1].strip())
            self.entries_task2["Кількість призначених продуктів:"].insert(0, lines[2].strip())
            self.entries_task2["Значення L:"].insert(0, lines[3].strip())
            self.entries_task2["Матриця виробництва (построчно, через коми):"].insert(0, lines[4].strip())
            self.entries_task2["Значення Y призначені (через коми):"].insert(0, lines[5].strip())
            self.entries_task2["Значення B (через коми):"].insert(0, lines[6].strip())
            self.entries_task2["Значення C матриця (через коми):"].insert(0, lines[7].strip())
            self.entries_task2["Значення F (через коми):"].insert(0, lines[8].strip())
            self.entries_task2["Значення пріоритетів (через коми):"].insert(0, lines[9].strip())
            self.entries_task2["Значення директивних строків(через коми):"].insert(0, lines[10].strip())
            self.entries_task2["Значення T_0 (через коми):"].insert(0, lines[11].strip())
            self.entries_task2["Значення Alpha (через коми):"].insert(0, lines[12].strip())

        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def load_from_file_task3(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Текстові файли", "*.txt")])
            if not file_path:
                return

            with open(file_path, 'r') as file:
                lines = file.readlines()

            self.entries_task3["Кількість агрегованих продуктів:"].insert(0, lines[0].strip())
            self.entries_task3["Кількість факторів виробництва:"].insert(0, lines[1].strip())
            self.entries_task3["Кількість призначених продуктів:"].insert(0, lines[2].strip())
            self.entries_task3["Значення L:"].insert(0, lines[3].strip())
            self.entries_task3["Матриця виробництва (построчно, через коми):"].insert(0, lines[4].strip())
            self.entries_task3["Значення Y призначені (через коми):"].insert(0, lines[5].strip())
            self.entries_task3["Значення B (через коми):"].insert(0, lines[6].strip())
            self.entries_task3["Значення C матриця (через коми):"].insert(0, lines[7].strip())
            self.entries_task3["Значення F (через коми):"].insert(0, lines[8].strip())
            self.entries_task3["Значення пріоритетів (через коми):"].insert(0, lines[9].strip())
            self.entries_task3["Значення директивних строків(через коми):"].insert(0, lines[10].strip())
            self.entries_task3["Значення T_0 (через коми):"].insert(0, lines[11].strip())
            self.entries_task3["Значення Alpha (через коми):"].insert(0, lines[12].strip())
            self.entries_task3["Значення Omega (через коми):"].insert(0, lines[13].strip())

        except Exception as e:
            messagebox.showerror("Помилка", str(e))

    def load_from_file_task4(self):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Текстові файли", "*.txt")])
            if not file_path:
                return

            with open(file_path, 'r') as file:
                lines = file.readlines()

            if len(lines) != 16:
                raise ValueError("Вхідний файл повинен містити рівно 16 рядків.")

            for key in self.entries_task4:
                self.entries_task4[key].delete(0, tk.END)

            self.entries_task4["Кількість агрегованих продуктів:"].insert(0, lines[0].strip())
            self.entries_task4["Кількість факторів виробництва:"].insert(0, lines[1].strip())
            self.entries_task4["Кількість призначених продуктів:"].insert(0, lines[2].strip())
            self.entries_task4["Значення L:"].insert(0, lines[3].strip())
            self.entries_task4["Значення M_L:"].insert(0, lines[4].strip())
            self.entries_task4["Матриця виробництва (построчно, через коми):"].insert(0, lines[5].strip())
            self.entries_task4["Значення Y призначені (через коми):"].insert(0, lines[6].strip())
            self.entries_task4["Значення B (через коми):"].insert(0, lines[7].strip())
            self.entries_task4["Значення C_L (через коми):"].insert(0, lines[8].strip())
            self.entries_task4["Значення F (через коми):"].insert(0, lines[9].strip())
            self.entries_task4["Значення пріоритетів (через коми):"].insert(0, lines[10].strip())
            self.entries_task4["Значення директивних строків(через коми):"].insert(0, lines[11].strip())
            self.entries_task4["Значення T_0 (через коми):"].insert(0, lines[12].strip())
            self.entries_task4["Значення Alpha (через коми):"].insert(0, lines[13].strip())
            self.entries_task4["Значення Omega (через коми):"].insert(0, lines[14].strip())
            self.entries_task4["Значення P_L (через коми):"].insert(0, lines[15].strip())

        except Exception as e:
            messagebox.showerror("Помилка", str(e))


def solve_production_problem_task1(production_data):
    """Defines and solves the linear programming problem for production optimization."""
    num_aggregated_products, num_production_factors, num_assigned_products, L, production_matrix, y_assigned, b, c, f, priorities, directive_terms, t_0, alpha, omega = production_data

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
    for i, omega_value in enumerate(omega):
        for l in range(num_production_factors):
            objective.SetCoefficient(y[l], c[i][l] * priorities[l] * omega_value)
        for l in range(num_assigned_products):
            objective.SetCoefficient(z[l], -f[l] * omega_value)
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
        num_assigned_products)], objective.Value()


def solve_production_problem_task2(production_data):
    """Defines and solves the linear programming problem for production optimization."""
    num_aggregated_products, num_production_factors, num_assigned_products, L, production_matrix, y_assigned, b, c, f, priorities, directive_terms, t_0, alpha = production_data

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
    for i, omega_value in enumerate(c):
        for l in range(num_production_factors):
            objective.SetCoefficient(y[l], omega_value[l] * priorities[l])
        for l in range(num_assigned_products):
            objective.SetCoefficient(z[l], -f[l])
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
        num_assigned_products)], objective.Value()


def solve_production_problem_task3(production_data):
    """Defines and solves the linear programming problem for production optimization."""
    num_aggregated_products, num_production_factors, num_assigned_products, L, production_matrix, y_assigned, b, c, f, priorities, directive_terms, t_0, alpha, omega = production_data

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
    for i, omega_value in enumerate(omega):
        for l in range(num_production_factors):
            objective.SetCoefficient(y[l], c[i][l] * priorities[l] * omega_value)
        for l in range(num_assigned_products):
            objective.SetCoefficient(z[l], -f[l] * omega_value)
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
        num_assigned_products)], objective.Value()


def solve_production_problem_task4(production_data):
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
    for l, omega_value in enumerate(omega):
        for m in range(M_L):
            for i in range(num_production_factors):
                objective.SetCoefficient(y[i], C_L[m][l][i] * priorities[i] * omega_value * P_L[m])
            for i in range(num_assigned_products):
                objective.SetCoefficient(z[i], -f[i] * P_L[m])
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
        num_assigned_products)], objective.Value()


def find_optimal_solution_task1(production_data, l):
    """Defines and solves the linear programming problem for production optimization."""
    num_aggregated_products, num_production_factors, num_assigned_products, _, production_matrix, y_assigned, b, c, f, priorities, directive_terms, t_0, alpha, _ = production_data

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
        objective.SetCoefficient(y[i], c[l][i] * priorities[i])
    for i in range(num_assigned_products):
        objective.SetCoefficient(z[i], -f[i])
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
        num_assigned_products)], objective.Value()


def find_optimal_solution_task2(production_data, l):
    """Defines and solves the linear programming problem for production optimization."""
    num_aggregated_products, num_production_factors, num_assigned_products, _, production_matrix, y_assigned, b, c, f, priorities, directive_terms, t_0, alpha = production_data

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
        objective.SetCoefficient(y[i], c[l][i] * priorities[i])
    for i in range(num_assigned_products):
        objective.SetCoefficient(z[i], -f[i])
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
        num_assigned_products)], objective.Value()


def find_optimal_solution_task3(production_data, l):
    """Defines and solves the linear programming problem for production optimization."""
    num_aggregated_products, num_production_factors, num_assigned_products, _, production_matrix, y_assigned, b, c, f, priorities, directive_terms, t_0, alpha, _ = production_data

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
        objective.SetCoefficient(y[i], c[l][i] * priorities[i])
    for i in range(num_assigned_products):
        objective.SetCoefficient(z[i], -f[i])
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
        num_assigned_products)], objective.Value()


def find_temp_optimal_solution(temp_production_data, num_production_factors, num_assigned_products,
                               num_aggregated_products):
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
    return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
        num_assigned_products)], objective.Value()


if __name__ == "__main__":
    root = tk.Tk()
    app = ProductionProblemApp(root)
    root.mainloop()