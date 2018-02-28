import cvxpy as cvx
# import networkx as nx
# import matplotlib.pyplot as plt
import numpy as np
from random import shuffle


def binarize_probabilities(mat, cash_vector=10000):
    """Turns a matrix of probabilities into a binary matrix.

    Args:
        mat (numpy ndarray): Probability matrix.

    Returns:
        A matrix of 1's and 0's.
    """
    # Total number of probabilities
    num_probs = mat.shape[0] * mat.shape[1]

    # Another probability matrix is generated and to determine 1 or 0 we...
    # probs = np.random.negative_binomial(1, .7, size=num_probs).reshape(mat.shape)
    probs = np.random.uniform(size=num_probs).reshape(mat.shape)
    # probs = np.zeros_like(mat)
    # for (i, capital) in enumerate(cash_vector):
    #     num = capital / 1000000.0
    #     probs[i, :] = np.random.power(a=num, size=mat.shape[0])

    # ... compare the generated probability against the given probability matrix
    # if it is less than, then the entry is a 1 otherwise it is a 0
    bin_mat = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            bin_mat[i, j] = 1 if probs[i, j] < mat[i, j] else 0
    return bin_mat


def distribute_liabilities(adj_matrix, total_liabilities):
    """Distributes cumulative liabilities across a matrix.

    Args:
        adj_matrix (numpy ndarray): Adjacency matrix.
        total_liabilities (numpy array): The total liability for each entity.

    Returns:
        A matrix with liabilities equally spread across the adjacency matrix's connections.
    """
    # Create the liability matrix
    size = adj_matrix.shape[0]
    liability_mat = np.zeros_like(adj_matrix)

    # Spread total liability equally among connections.
    for i, liability in enumerate(total_liabilities):
        conns = adj_matrix[i, :].sum()
        if conns == 0:
            continue
        avg_liability = liability / conns
        for j in range(size):
            liability_mat[i, j] = adj_matrix[i, j] * avg_liability
    return liability_mat


def make_connections(connectivity_vector):
    """Generates a probability matrix from the given connectivity vector.

    Args:
        connectivity_vector (numpy array): Vector of connections for each node.

    Returns:
        A probability matrix where each i,j entry is the probability that i and j are connected.
    """
    size = connectivity_vector.shape[0]
    connections = cvx.Variable(size, size)
    objective = cvx.Minimize(cvx.sum_entries(connections))

    constraints = [connections[i, i] == 1 for i in range(size)]
    for i, connection in enumerate(connectivity_vector):
        connection_constraint = cvx.sum_entries(connections[i, :]) + cvx.sum_entries(connections[:, i]) >= connection
        constraints.append(connection_constraint)

    for i in range(size):
        for j in range(size):
            lt_one_constraint = connections[i, j] <= 1
            constraints.append(lt_one_constraint)

    problem = cvx.Problem(objective, constraints)
    problem.solve()

    real_connections = connections.value
    return real_connections


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

        
class DeterministicRatioNetwork:
    
    def __init__(self, size, liabilities=None, recovery_rate=0.0, initial_cap=10000):
        self.size = size
        self.liabilities = liabilities if liabilities is not None else np.zeros((size, size))
        self.recovery_rate = recovery_rate
        self.initial_cap = initial_cap

    def reset_net(self):
        for i in range(self.size):
            for j in range(i + 1, self.size):
                self.liabilities[i, j] = max(self.liabilities[i, j] - self.liabilities[j, i], 0)
                self.liabilities[j, i] = max(self.liabilities[j, i] - self.liabilities[i, j], 0)

    def default(self, i):
        self.liabilities[i, i] = 0

    def recover(self, i):
        for j in range(self.size):
            if i != j:
                self.liabilities[j, j] += self.recovery_rate * self.liabilities[j, i]
                self.liabilities[j, i] = 0

    def step(self):
        # Select entry to add debt to
        rand_i = np.random.randint(self.size)
        rand_j = np.random.randint(self.size)
        rand_prop = 0.1

        # Add debt
        capital = self.liabilities[rand_i, rand_i]
        assets = self.liabilities[rand_i, :].sum() - capital
        liabilities = self.liabilities[:, rand_i].sum() - capital
        
        # TODO: what if we simplify by not growing loan by up to 10% of loan value
        # but by up to 10% of capital?
        # Should we subtract from capital when we're issuing another loan?
        if capital != 0:
            self.liabilities[rand_i, rand_j] += rand_prop * self.liabilities[rand_i, rand_i]
            self.liabilities[rand_i, rand_i] -= rand_prop * self.liabilities[rand_i, rand_i]
        else:
            if rand_i == rand_j:
                self.liabilities[rand_i, rand_j] = self.initial_cap

        # Settle
        ratios = np.zeros(self.size)
        previous_defaults = 0
        num_defaults = 0
        while True:  # Cascade until no more defaults
            for i in range(self.size):
                """
                capital = self.liabilities[i, i]
                assets = self.liabilities[i, :].sum()
                liabilities = self.liabilities[:, i].sum()
                net = capital + assets - liabilities
                if net < 0:
                    self.default(i)
                    self.recover(i)
                    num_defaults += 1
                """
                #"""Ratio cascade
                capital = self.liabilities[i, i]
                assets = self.liabilities[i, :].sum() - capital
                liabilities = self.liabilities[:, i].sum() - capital
                
                if liabilities != 0 and capital != 0:
                    ratios[i] = capital / liabilities
                    if capital / liabilities < 0.1:
                        self.default(i)
                        self.recover(i)
                        num_defaults += 1
                #"""
            if previous_defaults == num_defaults:
                break
            previous_defaults = num_defaults
        return ratios, num_defaults

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(self.liabilities, interpolation='nearest', cmap=plt.cm.hot)
        plt.colorbar()
        plt.show()

        
class TestNetwork:
    
    def __init__(self, size, liabilities=None, liability_ratio_mean=12, liability_ratio_sd=2, recovery_rate=0.0, initial_cap=10000, max_cap=10000, sampling_rate=0.2):
        self.size = size
        self.liabilities = liabilities if liabilities is not None else np.zeros((size, size))
        self.max_cap = max([self.liabilities[i, i] for i in range(size)])
        self.recovery_rate = recovery_rate
        self.initial_cap = initial_cap
        self.liability_ratio_mean = liability_ratio_mean
        self.liability_ratio_sd = liability_ratio_sd
        self.standing_banks = sum([1 if self.liabilities[i, i] > 0 else 0 for i in range(size)])
        self.sampling_rate = sampling_rate

    def reset_net(self):
        for i in range(self.size):
            for j in range(i + 1, self.size):
                self.liabilities[i, j] = max(self.liabilities[i, j] - self.liabilities[j, i], 0)
                self.liabilities[j, i] = max(self.liabilities[j, i] - self.liabilities[i, j], 0)

    def default(self, i):
        self.liabilities[i, i] = 0

    def recover(self, i):
        for j in range(self.size):
            if i != j:
                self.liabilities[j, j] += self.recovery_rate * self.liabilities[j, i]
                self.liabilities[j, i] = 0

    def step(self):
        # mask = np.ones(self.liabilities.shape, dtype=bool)
        # np.fill_diagonal(mask, 1)
        # print("Total capital: {0}".format(self.liabilities[mask].sum()))
        # print("Number banks: {0}".format(self.standing_banks))
        # max_ind = np.unravel_index(self.liabilities[mask].argmax(), self.liabilities.shape)
        # max_val = self.liabilities[mask].max()
        # print(max_val)
        # print("Cash for i: {0} Cash for j: {1} Asset for i {2} Liability for j {3}".format(
        #     self.liabilities[max_ind[0], max_ind[0]],
        #     self.liabilities[max_ind[1], max_ind[1]],
        #     self.liabilities[max_ind[0], max_ind[1]],
        #     self.liabilities[max_ind[1], max_ind[0]]
        # ))
        # Select entry to add debt to
        rand_i = np.random.randint(self.size)
        rand_j = np.random.randint(self.size)
        # Add debt
        capital = self.liabilities[rand_i, rand_i]
        assets = self.liabilities[rand_i, :].sum() - capital
        liabilities = self.liabilities[:, rand_i].sum() - capital
        # print(capital, rand_i)
        if not np.isclose(capital, 0):
            num = max((1 - capital / self.max_cap), 0.2)
            rand_prop = np.random.power(a=num, size=1)[0]
            # print(num, rand_prop)
            while True:
                if self.standing_banks <= 1:
                    break
                if np.isclose(self.liabilities[rand_j, rand_j], 0):
                    rand_j = np.random.randint(self.size)
                else:
                    break
            self.liabilities[rand_i, rand_j] += rand_prop * self.liabilities[rand_i, rand_i]
            self.liabilities[rand_i, rand_i] -= rand_prop * self.liabilities[rand_i, rand_i]
        else:
            # if np.isclose(rand_i, rand_j):
            #     self.liabilities[rand_i, rand_j] = self.initial_cap
            #     self.standing_banks += 1
            self.liabilities[rand_i, rand_i] = self.initial_cap
            self.standing_banks += 1

        # Settle
        results = {}
        ratios = np.zeros(self.size)
        previous_defaults = 0
        num_defaults = 0
        defaulted_banks = []
        rand_inds_to_check = np.random.choice(range(self.size), 1, replace=False)
        rand_inds_to_check = rand_inds_to_check[: int(self.sampling_rate * self.size)]
        num_should_default = 0
        for i in range(self.size):
            capital = self.liabilities[i, i]
            assets = self.liabilities[i, :].sum() - capital
            liabilities = self.liabilities[:, i].sum() - capital
            net = capital + assets - liabilities
            if net < 0:
                num_should_default += 1
                if i in rand_inds_to_check:
                    defaulted_banks.append(i)
                    num_defaults += 1
        results['ratios'] = ratios
        results['ratio_defaults'] = num_defaults
        previous_default = 0
        num_defaults = 0
        while True:  # Cascade until no more defaults
            for i in range(self.size):
                if i in defaulted_banks:
                    continue
                capital = self.liabilities[i, i]
                exposures = 0
                for defaulted_bank in defaulted_banks:
                    exposures += self.liabilities[i, defaulted_bank]
                    if capital < exposures:
                        defaulted_banks.append(i)
                        num_defaults += 1
                        num_should_default += 1
                        break
            if previous_defaults == num_defaults:
                break
            previous_defaults = num_defaults
        for default in defaulted_banks:
            self.liabilities[:, default] = 0
            self.liabilities[default, :] = 0
        results['cascade_defaults'] = num_defaults
        self.standing_banks -= (results['ratio_defaults'] + results['cascade_defaults'])
        # print("Number standing: {0} Number defaults (actual): {1} Number defaults (perfect): {2}".format(
        #     self.standing_banks,
        #     (results['ratio_defaults'] + results['cascade_defaults']),
        #     num_should_default))
        return results

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(self.liabilities, interpolation='nearest', cmap=plt.cm.hot)
        plt.colorbar()
        plt.show()

        
class DeterministicNetwork:
    
    def __init__(self, size, liabilities=None, recovery_rate=0.0):
        self.size = size
        self.liabilities = liabilities if liabilities is not None else np.zeros((size, size))
        self.recovery_rate = recovery_rate

    def reset_net(self):
        for i in range(self.size):
            for j in range(i + 1, self.size):
                self.liabilities[i, j] = max(self.liabilities[i, j] - self.liabilities[j, i], 0)
                self.liabilities[j, i] = max(self.liabilities[j, i] - self.liabilities[i, j], 0)

    def default(self, i):
        self.liabilities[i, i] = 0

    def recover(self, i):
        for j in range(self.size):
            if i != j:
                self.liabilities[j, j] += self.recovery_rate * self.liabilities[j, i]
                self.liabilities[j, i] = 0

    def step(self):
        for i in range(self.size):
            capital = self.liabilities[i, i]
            assets = self.liabilities[i, :].sum()
            liabilities = self.liabilities[:, i].sum()
            net = capital + assets - liabilities
            print(i, assets, liabilities, capital, net)
            if net < 0:
                self.default(i)
                self.recover(i)

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')
        plt.imshow(self.liabilities, interpolation='nearest', cmap=plt.cm.hot)
        plt.colorbar()
        plt.show()
