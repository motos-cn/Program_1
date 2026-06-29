import numpy as np
from .smiles2graph import smilesToGraphConverter


class GraphProcessor:

    def __init__(self):
        self.converter = smilesToGraphConverter()

    def process_ionic_liquid(self, il_smiles, temperature, pressure,
                              t_mean=None, t_std=None,
                              p_mean=None, p_std=None):
        graph = self.converter.smiles2graph(il_smiles)
        if t_mean is not None and t_std is not None and t_std > 0:
            temperature = (temperature - t_mean) / t_std
        # Pressure: log-transform then standardize
        pressure = np.log(pressure)
        if p_mean is not None and p_std is not None and p_std > 0:
            pressure = (pressure - p_mean) / p_std
        return [graph, temperature, pressure]
