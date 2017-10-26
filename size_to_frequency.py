"""This script will run the avalanche model and save the size-to-frequency as a JSON file where the key is the cascade size and value is the number of occurences."""

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

    defaults_to_freq = {}

    for z in tqdm(range(1000000)):
        model = TestNetwork(100, mat)
        model.reset_net()

        step_result = model.step()
        defaults = step_result['cascade_defaults'] + step_result['ratio_defaults']
        if defaults in defaults_to_freq:
            defaults_to_freq[defaults] += 1
        else:
            defaults_to_freq[defaults] = 1
    with open('result_{0}.json'.format(k), 'w') as fp:
        json.dump(defaults_to_freq, fp)
