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
    "CardinalityNodePruning": CardinalityNodePruning,
    "ReciprocalCardinalityNodePruning": ReciprocalCardinalityNodePruning,
    "ReciprocalWeightedNodePruning" :  ReciprocalWeightedNodePruning,
    "WeightedEdgePruning" :  WeightedEdgePruning,
    "WeightedNodePruning" :  WeightedNodePruning
}


if __name__ == '__main__':

    # Read the JSON file
    with open("data/candidate_pairs/standard_blocking/.config.json", 'r') as file:
        data_json = json.load(file)


        results  = []
        for dataset in ['D2','D3', 'D5','D6', 'D7', 'D8']:
            result = {}

            params = data_json.get(dataset, {})
            print(params)

            cp = pd.read_csv(f'data/candidate_pairs/original/{dataset}.csv')
            cp_columns = list(cp.columns)
            clean_files = [cl.replace("clean", "").replace(".csv", "") for cl in cp_columns]
            

            sep = '|' if dataset!='D3' else '#'

            d1 = pd.read_csv(f"data/{dataset}/{clean_files[0]}clean.csv", sep=sep)
            d2 = pd.read_csv(f"data/{dataset}/{clean_files[1]}clean.csv", sep=sep)
            gt = pd.read_csv(f"data/{dataset}/gtclean.csv", sep=sep)
            
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
                bp = BlockPurging(smoothing_factor=params['smoothing_factor'])
                blocks = bp.process(blocks, data)

            bf = BlockFiltering(ratio = getRatio(params['block_filtering']))
            print(bf.ratio)
            input("Press Enter to continue...")

            blocks = bf.process(blocks, data)


            if "weighting_scheme" in params: 
                cc = cc_dict[params['comparison_cleaning']](params['weighting_scheme'])
            else:
                cc = cc_dict[params['comparison_cleaning']]()
            blocks = cc.process(blocks, data)

            print(dataset)

            ev = cc.evaluate(blocks)

            print(ev)

            input("Press Enter to continue...")

            # if ev['Recall %'] > 90:
            cp_df = cc.export_to_df(blocks)
            cp_df.columns = cp_columns
            cp_df.to_csv(f"data/candidate_pairs/standard_blocking/{dataset}.csv", index=False)
            
            results.append({
                "dataset" : dataset,
                "candidate_pair" : 'Standard-Blocking',
                "total_candidate_pairs" : len(cp_df),
                'precision': ev['Precision %'],
                'recall' : ev['Recall %'] ,
                'f1' : ev["F1 %"]
            })
        results_df = pd.DataFrame(results)
        results_df.to_csv("data/candidate_pairs/candidate_pairs_perforamnce.csv", header=False, index=False, mode='a+')
        








            