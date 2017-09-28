import json
from tqdm import tqdm
import numpy as np
import cvxpy as cvx
from contagion import binarize_probabilities, distribute_liabilities, make_connections, DeterministicRatioNetwork, TestNetwork


for k in range(30):
    cash_vector = np.random.normal(10000, 10000, 100)
    cash_vector[cash_vector <= 0] = 1*10**-10
    # cash_vector[cash_vector > 5000] = 6500
    cash_to_connectivity = lambda x: np.log(x).astype(int)
    connectivity_vector = cash_to_connectivity(cash_vector)

    # Make the adjacency matrix
    mat = make_connections(connectivity_vector)
    mat = binarize_probabilities(mat)

    # Distribute liabilities
    leverage_ratios = np.random.normal(10, 2, 100)
    leverage_ratios[leverage_ratios < 5] = 5

    liabilities = np.multiply(cash_vector, leverage_ratios)
    mat = distribute_liabilities(mat, liabilities)
    for i, cash in enumerate(cash_vector):
        mat[i, i] = cash
    defaults = np.zeros((1000000, 2))
    for z in tqdm(range(1000000)):
        model = TestNetwork(100, mat)
        model.reset_net()

        results = model.step()
        defaults[z, 0] = results['ratio_defaults']
        defaults[z, 1] = results['cascade_defaults']
    np.savetxt('defaults_{0}.csv'.format(k), defaults, delimiter=',')
