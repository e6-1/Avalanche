import cvxpy as cvx
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle


def make_connections(connectivity_vector, randomize=False):
    # Get the number of variables
    size = connectivity_vector.shape[0]

    # The matrix variable for the optimization problem
    connections = cvx.Variable(size, size)

    # Minimize the total sum of the connections variable over all i, j
    objective = cvx.Minimize(cvx.sum_entries(connections))

    # Force an identity matrix
    constraints = [connections[i, i] == 1 for i in range(size)]

    # Iterate through and add the following constraint:
    # sum_j M_{ij} + sum_j M_{ji} >= k_i, for all i
    for i, connection in enumerate(connectivity_vector):
        connection_constraint = cvx.sum_entries(connections[i, :]) + cvx.sum_entries(connections[:, i]) >= connection
        constraints.append(connection_constraint)

    # Constrain each connection to be real-valued, but less than or equal to 1
    for i in range(size):
        for j in range(size):
            lt_one_constraint = connections[i, j] <= 1
            constraints.append(lt_one_constraint)

    # Solve the optimization problem, given the objective and constraints
    problem = cvx.Problem(objective, constraints)
    problem.solve()

    # This last chunk will convert the real-valued matrix from the optimization into a binary matrix
    real_connections = connections.value
    adj_mat = np.zeros((size, size))
    inds = range(size)

    # If the randomization flag is set, then randomly select rows to binarize
    if randomize:
        shuffle(inds)

    # Go through, and pick the top `n` connections to be 1s and the rest to be 0s
    for i in inds:
        connection = connectivity_vector[i]
        connection = max(0, int(connection - adj_mat[i, :].sum()))
        max_connection_inds = real_connections[i].argsort()[::-1]
        max_connection_inds = max_connection_inds[0, :connection]
        for j in max_connection_inds:
            adj_mat[i, j] = 1
            adj_mat[j, i] = -1
    return adj_mat


class ContagionNetwork:
    def __init__(self, exposure_matrix, capital_ratios, defaults=[]):
        self.exposure_matrix = exposure_matrix
        self.capital_ratios = capital_ratios
        self.defaults = defaults

    def step(self):
        next_defaults = []
        for institution, exposures in enumerate(self.exposure_matrix):
            assets = exposures.sum()
            for default in self.defaults:
                capital = self.capital_ratios[institution] * assets
                if capital < exposures[default]:
                    next_defaults.append(institution)
                    break
        self.defaults += next_defaults
