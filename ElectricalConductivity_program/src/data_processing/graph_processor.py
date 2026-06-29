import numpy as np
from .smiles2graph import smilesToGraphConverter


class GraphProcessor:

    def __init__(self):
        self.converter = smilesToGraphConverter()

    def process_ionic_liquid(self, il_smiles, temperature, t_mean=None, t_std=None):
        graph = self.converter.smiles2graph(il_smiles)
        if t_mean is not None and t_std is not None and t_std > 0:
            temperature = (temperature - t_mean) / t_std
        return [graph, temperature]
