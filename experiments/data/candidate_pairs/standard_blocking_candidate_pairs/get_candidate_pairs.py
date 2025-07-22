import pandas as pd
from pyjedai.block_building import StandardBlocking
from pyjedai.block_cleaning import BlockFiltering, BlockPurging
import json
from pyjedai.comparison_cleaning import * 


def getRatio(iteration_number):
    return 0.025 + iteration_number * 0.025


# def getWeight(iteration_number):
#     i = 0 + iteration_number * 1

# ws = [
#     'CNC',
#     'CBS',
#     'ECBS',
#     'JS',
#     'EJS',
#     "X2"
# ]


cc_dict = {
    "BLAST" : BLAST,
    "CardinalityEdgePruning": CardinalityEdgePruning,
    "CardinalityNodePruning": CardinalityNodePruning(),
    "ReciprocalCardinalityNodePruning": ReciprocalCardinalityNodePruning(),
    "ReciprocalWeightedNodePruning" :  ReciprocalWeightedNodePruning(),
    "WeightedEdgePruning" :  WeightedEdgePruning(),
    "WeightedNodePruning" :  WeightedNodePruning()
}


if __name__ == '__main__':

    # Read the JSON file
    with open('standard_blocking_candidate_pairs/.config.json', 'r') as file:
        data_json = json.load(file)



        for dataset in ['D3']:

            params = data_json.get(dataset, {})
            print(params)

            cp = pd.read_csv(f'candidate_pairs/{dataset}.csv')
            cp_columns = list(cp.columns)
            clean_files = [cl.replace("clean", "").replace(".csv", "") for cl in cp_columns]
            

            sep = '|' if dataset!='D3' else '#'

            d1 = pd.read_csv(f"data_clean/{dataset}/{clean_files[0]}clean.csv", sep=sep)
            d2 = pd.read_csv(f"data_clean/{dataset}/{clean_files[1]}clean.csv", sep=sep)
            gt = pd.read_csv(f"data_clean/{dataset}/gtclean.csv", sep=sep)
            
            data = Data(
                dataset_1=d1,
                id_column_name_1='id',
                id_column_name_2='id',
                dataset_2=d2,
                ground_truth=gt
            )

            standard_blocking = StandardBlocking()
            blocks = standard_blocking.build_blocks(data)

            if params['block_purging']:
                bp = BlockPurging()
                blocks = bp.process(blocks, data)

            bf = BlockFiltering(ratio = getRatio(params['block_filtering']))
            blocks = bf.process(blocks, data)


            if "weighting_scheme" in params: 
                cc = cc_dict[params['comparison_cleaning']](params['weighting_scheme'])
            else:
                cc = cc_dict[params['comparison_cleaning']]
            blocks = cc.process(blocks, data)

            print(dataset)

            ev = cc.evaluate(blocks)

            if ev['Recall %'] > 90:
                cp_df = cc.export_to_df(blocks)
                cp_df.columns = cp_columns
                cp_df.to_csv(f"standard_blocking_candidate_pairs/{dataset}.csv", index=False)






            