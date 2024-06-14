import random
import solver
import numpy as np
from ortools.linear_solver import pywraplp

random.seed(1810)
np.random.seed(1810)

# Constants
num_aggregated_products = 20  # m
num_production_factors = 10  # n
num_assigned_products = 9  # n1
L = 5


def generate_production_data():
    """Generates production data including matrix, assigned quantities, resource limits, etc."""
    production_matrix = np.random.uniform(0.1, 1, (num_aggregated_products, num_production_factors))
    y_assigned = [random.uniform(1, 100) for _ in range(num_assigned_products)]
    b = [random.uniform(y_assigned[i] if i < num_assigned_products else 1000, 10000) for i in
         range(num_aggregated_products)]
    c = [[random.uniform(1, 10) for _ in range(num_production_factors)] for _ in range(L)]
    priorities = np.ones(num_production_factors)
    directive_terms = sorted([random.uniform(10, 100) for _ in range(num_production_factors)])
    t_0 = [float(i) for i in range(num_production_factors)]
    alpha = [random.uniform(1.0, 2) for _ in range(num_production_factors)]
    omega = [random.uniform(0, 1) for _ in range(L)]
    omega = [np.exp(omega_i) / sum(np.exp(omega)) for omega_i in omega]  # Softmax normalization
    a_plus = [random.uniform(0, 1) for _ in range(num_assigned_products)]
    a_plus = [np.exp(a_plus_i) / sum(np.exp(a_plus)) for a_plus_i in a_plus]
    a_minus = [random.uniform(0, 1) for _ in range(num_assigned_products)]
    a_minus = [np.exp(a_minus_i) / sum(np.exp(a_minus)) for a_minus_i in a_minus]
    return production_matrix, y_assigned, b, c, priorities, directive_terms, t_0, alpha, omega, a_plus, a_minus


def print_data(data, def_names=(
        "Production matrix", "Y assigned", "B", "C", "F", "Priorities", "Directive terms", "T_0", "Alpha", "Omega",
        "A_plus", "A_minus")):
    """Prints the generated production data in a formatted way."""
    for name, value in zip(def_names, data):
        print(f"{name}:\n{np.round(value, 2)}")
        print("=" * 100)


def find_temp_optimal_solution(production_data, l):
    """Finds the temporary optimal solution for the given production data."""
    production_matrix, y_assigned, b, c, priorities, directive_terms, t_0, alpha, _, a_plus, a_minus = production_data

    lp_solver = pywraplp.Solver.CreateSolver("GLOP")

    # Define Variables
    y = [lp_solver.NumVar(0, lp_solver.infinity(), f"y_{i}") for i in range(num_production_factors)]
    u_plus = [lp_solver.NumVar(0, lp_solver.infinity(), f"U_plus_{i}") for i in range(num_assigned_products)]
    u_minus = [lp_solver.NumVar(0, lp_solver.infinity(), f"U_minus_{i}") for i in range(num_assigned_products)]

    # Define Constraints
    for i in range(num_aggregated_products):
        lp_solver.Add(sum(production_matrix[i][j] * y[j] for j in range(num_production_factors)) <= b[i])
    for i in range(num_assigned_products):
        lp_solver.Add(u_plus[i] >= 0)
        lp_solver.Add(u_minus[i] >= 0)
        lp_solver.Add(directive_terms[i] - (t_0[i] + alpha[i] * y[i]) <= u_plus[i] - u_minus[i])
    for i in range(num_assigned_products):
        lp_solver.Add(y[i] >= y_assigned[i])

    # Define Goal
    objective = lp_solver.Objective()
    for i in range(num_production_factors):
        objective.SetCoefficient(y[i], c[l][i] * priorities[i])
    for i in range(num_assigned_products):
        objective.SetCoefficient(u_plus[i], -a_plus[i])
        objective.SetCoefficient(u_minus[i], -a_minus[i])
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(num_production_factors)], [u_plus[i].solution_value() for i in
                                                                            range(num_assigned_products)], [
        u_minus[i].solution_value() for i in range(num_assigned_products)], objective.Value()


def solve_production_problem(production_data):
    """Defines and solves the linear programming problem for production optimization."""
    production_matrix, y_assigned, b, c, priorities, directive_terms, t_0, alpha, omega, a_plus, a_minus = production_data

    lp_solver = pywraplp.Solver.CreateSolver("GLOP")

    # Define Variables
    y = [lp_solver.NumVar(0, lp_solver.infinity(), f"y_{i}") for i in range(num_production_factors)]
    u_plus = [lp_solver.NumVar(0, lp_solver.infinity(), f"U_plus_{i}") for i in range(num_assigned_products)]
    u_minus = [lp_solver.NumVar(0, lp_solver.infinity(), f"U_minus_{i}") for i in range(num_assigned_products)]

    # Define Constraints
    for i in range(num_aggregated_products):
        lp_solver.Add(sum(production_matrix[i][j] * y[j] for j in range(num_production_factors)) <= b[i])
    for i in range(num_assigned_products):
        lp_solver.Add(u_plus[i] >= 0)
        lp_solver.Add(u_minus[i] >= 0)
        lp_solver.Add(directive_terms[i] - (t_0[i] + alpha[i] * y[i]) <= u_plus[i] - u_minus[i])
    for i in range(num_assigned_products):
        lp_solver.Add(y[i] >= y_assigned[i])

    objective = lp_solver.Objective()
    for i, o in enumerate(omega):
        for l in range(num_production_factors):
            objective.SetCoefficient(y[l], c[i][l] * priorities[l] * o)
        for l in range(num_assigned_products):
            objective.SetCoefficient(u_plus[l], -a_plus[l])
            objective.SetCoefficient(u_minus[l], -a_minus[l])
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(num_production_factors)], [u_plus[i].solution_value() for i in
                                                                            range(num_assigned_products)], [
        u_minus[i].solution_value() for i in range(num_assigned_products)], objective.Value()


if __name__ == "__main__":
    test_production_data = generate_production_data()
    F_optimums = [find_temp_optimal_solution(test_production_data, l)[3] for l in range(L)]
    print_data(test_production_data)
    y_solution, u_plus_solution, u_minus_solution, objective_value = solve_production_problem(test_production_data)
    t_0 = test_production_data[6]
    alpha = test_production_data[7]
    policy_deadlines = [t_0[i] + alpha[i] * y_solution[i] for i in range(num_assigned_products)]
    completion_dates = [u_plus_solution[i] - u_minus_solution[i] for i in range(num_assigned_products)]
    differences = [policy_deadlines[i] - completion_dates[i] for i in range(num_assigned_products)]
    print("Detailed results:")
    names = ["Objective", "Y_solution", "U_plus_solution", "U_minus_solution", "Policy deadlines", "Completion dates",
             "Differences"]
    print_data([objective_value, y_solution, u_plus_solution, u_minus_solution, policy_deadlines, completion_dates,
                differences], names)
    print("Differences between f_optimum and f_solution:")
    for l in range(L):
        # C_L^T * y_solution - (A_plus^T * U_plus_solution + A_minus^T * U_minus_solution)
        c_l = test_production_data[3][l]
        a_plus = test_production_data[9]
        a_minus = test_production_data[10]
        f_solution = sum([c_l[j] * y_solution[j] for j in range(num_production_factors)]) - sum(
            [a_plus[j] * u_plus_solution[j] for j in range(num_assigned_products)]) - sum(
            [a_minus[j] * u_minus_solution[j] for j in range(num_assigned_products)])
        difference = F_optimums[l] - f_solution
        omega_l = test_production_data[-3][l]
        print(f"{l = },\t{omega_l = :.2f},\t{F_optimums[l] = :.2f},\t{f_solution = :.2f},\t{difference = :.2f}")
