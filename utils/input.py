from typing import Any, List, Tuple, Dict
import numpy as np
import copy
from copy import deepcopy


def interface_input(default_config: Dict, input_data: Dict) -> Dict:
    new_default_config = deepcopy(default_config)
    if input_data["selected_assets"][0] != "Default Settings":
        nb_asset = len(input_data["selected_assets"])+2
        new_default_config["symbols"] = (input_data["selected_assets"] +
                                         default_config["symbols"][-1])
        index_And_weights = [[0 for _ in range(nb_asset)] for _ in range(2)]
        index_And_weights[0][0, -1] = [1, 1]
        new_default_config["portfolio"]["index_And_weights"] = index_And_weights

        id_matrix = np.identity(nb_asset).tolist()
        new_default_config["portfolio"]["ref_portfolios"] = {
            input_data["selected_assets"][i]: {
                                          "capital": None,
                                          "next_weights": id_matrix[i+1],
                                          "weights": None
                                        } for i in range(nb_asset-1)}
        eq_weight = 1/(nb_asset-1)
        new_default_config["portfolio"]["ref_portfolios"]["Eq_weighted"] = {
                                  "capital": None,
                                  "next_weights": [0.0 if (j == 0 or j == nb_asset-1)
                                                   else eq_weight for j in range(nb_asset)],
                                  "weights": None}
    #if input_data["selected_strategy"] != "Default Settings":

    return new_default_config
