import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# Hello
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
