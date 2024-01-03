import pandas as pd
import numpy as np

# use custom ollama model "worker" on llama_index
from llama_index.llms import Ollama
llm = Ollama(model='worker')

# paths for datasets, candidate pairs and groundtruth files
dataset_1 = 'dt_cp_gt/dt1_rest1.csv'
dataset_2 = 'dt_cp_gt/dt1_rest2.csv'
candidate_pairs = 'dt_cp_gt/cp1.csv'
groundtruth = 'dt_cp_gt/gt1.csv'

# read the files 
dt1 = pd.read_csv(dataset_1, sep='|')
dt2 = pd.read_csv(dataset_2, sep='|')
cp = pd.read_csv(candidate_pairs)
gt = pd.read_csv(groundtruth)

# convert to numpy arrays
dt1 = dt1.to_numpy()
dt2 = dt2.to_numpy()
cp = cp.to_numpy()
gt = gt.to_numpy()

# cut the indexes
dt1 = dt1[:, 1:]
dt2 = dt2[:, 1:]

# concatenate the strings in each column to a single string, omitting empty elements
dt1 = np.array([' '.join([x for x in row if isinstance(x, str)]) for row in dt1])
dt2 = np.array([' '.join([x for x in row if isinstance(x, str)]) for row in dt2])



for i in range(20):
    dt1_index = cp[i][0]
    dt2_index = cp[i][1]

    r1 = dt1[dt1_index]
    r2 = dt2[dt2_index]

    print(f"candidate pair {i}")
    print(f"record 1: {r1}")
    print(f"record 2: {r2}")

    query = f"record 1: {r1}, record 2: {r2}"

    resp = llm.complete(query)

    print(f"worker's response: {resp}")
    print(" ")