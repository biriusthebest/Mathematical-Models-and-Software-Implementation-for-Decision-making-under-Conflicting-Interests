import random
import solver
import numpy as np
from ortools.linear_solver import pywraplp

np.set_printoptions(linewidth=1000)

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
    f = [random.uniform(0.1, 1) for _ in range(num_assigned_products)]
    priorities = np.ones(num_production_factors)
    directive_terms = sorted([random.uniform(1, 100) for _ in range(num_production_factors)])
    t_0 = [float(i) for i in range(num_production_factors)]
    alpha = [random.uniform(1.0, 2) for _ in range(num_production_factors)]
    omega = [random.uniform(0, 1) for _ in range(L)]
    omega = [np.exp(omega_i) / sum(np.exp(omega)) for omega_i in omega]  # Softmax normalization
    return production_matrix, y_assigned, b, c, f, priorities, directive_terms, t_0, alpha, omega


def print_data(data, def_names=(
        "Production matrix", "Y assigned", "B", "C", "F", "Priorities", "Directive terms", "T_0", "Alpha", "Omega")):
    """Prints the generated production data in a formatted way."""
    for name, value in zip(def_names, data):
        print(f"{name}:\n{np.round(value, 2)}")
        print("=" * 100)


def find_optimal_solution(production_data, l):
    """Defines and solves the linear programming problem for production optimization."""
    production_matrix, y_assigned, b, c, f, priorities, directive_terms, t_0, alpha, _ = production_data

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


def solve_production_problem(production_data):
    """Defines and solves the linear programming problem for production optimization."""
    production_matrix, y_assigned, b, c, f, priorities, directive_terms, t_0, alpha, omega = production_data

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
    for i, o in enumerate(omega):
        for l in range(num_production_factors):
            objective.SetCoefficient(y[l], c[i][l] * priorities[l] * o)
        for l in range(num_assigned_products):
            objective.SetCoefficient(z[l], -f[l] * o)
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
        num_assigned_products)], objective.Value()


if __name__ == "__main__":
    test_production_data = generate_production_data()
    print_data(test_production_data)
    y_solution, z_solution, objective_value = solve_production_problem(test_production_data)
    t_0 = test_production_data[7]
    alpha = test_production_data[8]
    policy_deadlines = [t_0[i] + alpha[i] * y_solution[i] for i in range(num_assigned_products)]
    completion_dates = [z_solution[i] for i in range(num_assigned_products)]
    differences = [policy_deadlines[i] - completion_dates[i] for i in range(num_assigned_products)]
    print("Detailed results:")
    names = ["Objective", "Y_solution", "Z_solution", "Policy deadlines", "Completion dates", "Differences"]
    print_data([objective_value, y_solution, z_solution, policy_deadlines, completion_dates, differences], names)
    print("Differences between f_optimum and f_solution:")
    for l in range(L):
        optimum_value = find_optimal_solution(test_production_data, l)[2]
        f_solution = sum(test_production_data[3][l][i] * y_solution[i] for i in range(num_production_factors)) - sum(
            test_production_data[4][i] * z_solution[i] for i in range(num_assigned_products))
        difference = optimum_value - f_solution
        print(f"{l = },\t{optimum_value = :.2f},\t{f_solution = :.2f},\t{difference = :.2f}")
