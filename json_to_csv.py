# Example:
# > python json_to_csv.py NormalDistResults/

import os, json
import pandas as pd
import numpy as np
import sys

path_to_json = sys.argv[1]
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

for i, json_file in enumerate(json_files):
    with open(path_to_json + json_file) as f:
        data = json.load(f)
        arr = np.zeros((100, 1))
        for k in data:
            arr[int(k)] = data[k]
        np.savetxt(path_to_json + "result{0}.csv".format(i), arr)
