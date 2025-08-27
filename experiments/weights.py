import pandas as pd
import json
from examples_blocking import examples_dict_list
import os
from pyjedai.comparison_cleaning import * 
from pyjedai.block_building import *
from pyjedai.block_cleaning import * 
from pyjedai.joins import TopKJoin
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding


cc_dict = {
    "BLAST" : BLAST,
    "CardinalityEdgePruning": CardinalityEdgePruning,
    "CardinalityNodePruning": CardinalityNodePruning,
    "ReciprocalCardinalityNodePruning": ReciprocalCardinalityNodePruning,
    "ReciprocalWeightedNodePruning" :  ReciprocalWeightedNodePruning,
    "WeightedEdgePruning" :  WeightedEdgePruning,
    "WeightedNodePruning" :  WeightedNodePruning
}

def getRatio(iteration_number):
    return 0.025 + iteration_number * 0.025


DICE_SIM = 'dice'
COSINE_SIM = 'cosine'

CHARACTER_FOURGRAMS_MULTISET = 'qgrams_multiset'


K = [1, 4, 26, 1, 1, 1, 1, 2, 1, 5]
reversed_bool = [True, False, True, False, False, False, False, True, True, True]
similarity = [ DICE_SIM, COSINE_SIM, COSINE_SIM, COSINE_SIM,
            COSINE_SIM, COSINE_SIM, COSINE_SIM, COSINE_SIM,
            COSINE_SIM, COSINE_SIM ]

tokenizer = ['qgrams_multiset', 'qgrams_multiset','qgrams_multiset',
            'qgrams_multiset', 'qgrams', 'qgrams', 'qgrams', 
            'qgrams_multiset', 'qgrams', 'qgrams']

qgrams = [4, 3, 5, 2, 5, 5, 5, 4, 4, 4]



index = [2, 5, 6, 7, 8]


if __name__ == '__main__': 

    # # Read the JSON file
    # with open("data/candidate_pairs/standard_blocking/.config.json", 'r') as file:
    #     data_json = json.load(file)

    for i in index:
        
        dataset = f'D{i}'
        # params = data_json.get(dataset, {})
        
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
        emb = EmbeddingsNNBlockBuilding(vectorizer='sdistilroberta', similarity_search='faiss')
        blocks, g = emb.build_blocks(data, with_entity_matching=True)
        
        # join = TopKJoin(K=K[i-1], metric=similarity[i-1], tokenization=tokenizer[i-1], qgrams=qgrams[i-1])
        # g = join.fit(data, reverse_order=reversed_bool[i-1])
        
        # standard_blocking = StandardBlocking()
        # blocks = standard_blocking.build_blocks(data)

        # if params['block_purging']:
        #     bp = BlockPurging(smoothing_factor=params['smoothing_factor'])
        #     blocks = bp.process(blocks, data)

        # bf = BlockFiltering(ratio = getRatio(params['block_filtering']))
        # print(bf.ratio)

        # blocks = bf.process(blocks, data)


        # if "weighting_scheme" in params: 
        #     cc  = cc_dict[params['comparison_cleaning']](params['weighting_scheme'])
        # else:
        #     cc = cc_dict[params['comparison_cleaning']]()
        # blocks = cc.process(blocks, data)
        for candidate_pairs in ['original', 'standard_blocking']:
            results_df = pd.read_csv(f'results/{candidate_pairs}/{dataset}.csv')

            for _, row in results_df.iterrows():
                llm = row['model']
                if  "U" in llm or "∩" in llm: 
                    llm = llm.replace(" U ", "_union_") if "U" in llm else llm.replace(" ∩ ", "_intersection_") 

                examples = row['examples']
                responses_path = f'responses/{candidate_pairs}/{dataset}/{dataset}_{llm}_{examples}.csv'
                    
                if '-z' in llm and not os.path.exists(responses_path):
                    responses_path = f'responses/{candidate_pairs}/{dataset}/{dataset}_{llm}.csv'
                        
                responses_df = pd.read_csv(responses_path)
                
                if 's-weight' not in list(responses_df.columns):
                    # if len(list(responses_df.columns)) >= 4:
                    #     col = list(responses_df.columns)[0]
                    #     if col != 'D1': 
                    #         responses_df = responses_df.drop(columns=[col])

                    weights = []
                    for _, (id1, id2, _, _) in responses_df.iterrows():
                        block_id1 = data._ids_mapping_1[str(id1)]
                        block_id2 = data._ids_mapping_2[str(id2)]
                        
                        if g.has_edge(block_id1, block_id2): 
                            weights.append(g[block_id1][block_id2]['weight'])
                        elif g.has_edge(block_id2, block_id1):
                            weights.append(g[block_id2][block_id1]['weight'])
                        else:
                            weights.append(0.0)
                        
                    #     if (block_id1 in blocks and  block_id2 in blocks[block_id1]) or \
                    #         (block_id2 in blocks and block_id1 in blocks[block_id2]):
                    #             weights.append(cc._get_weight(block_id1, block_id2)) 
                    #     else: 
                    #         weights.append(0.0)
                    # max_weight = max(weights)
                    # while max_weight > 1.0: 
                    #     weights = [w/10.0 for w in weights]
                    #     max_weight = max(weights)
                    
                    responses_df['s-weight'] = weights
                    responses_df.to_csv(responses_path, index=False, mode='w+')
            print(f"{dataset} {candidate_pairs} DONE")
                    