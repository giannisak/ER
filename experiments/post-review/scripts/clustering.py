
from pyjedai.clustering import UniqueMappingClustering
import pandas as pd
from pyjedai.datamodel import Data
import os
from tqdm.auto import  tqdm
from networkx import Graph
from gpt_utils import _load_dataset, _get_responses_df

if  __name__ == '__main__':


    BLOCKING_TYPE = 'original'
    DIR = 'D5'
    POST_REVIEW_EXP = "movies"
    dt1_df, dt2_df, cp_df, gt_df, clean_files = _load_dataset(BLOCKING_TYPE, DIR)
    wef = {
        'original':{
            'weight': 'TopKJoin',
            's-weight': 'distilroberta'
        },
       'standard_blocking':{
           'weight': 'MetaBlocking-Method',
           's-weight': 'distilroberta'
       },
    }

    data = Data(
        dataset_1=dt1_df,
        id_column_name_1='id',
        id_column_name_2='id',
        dataset_2=dt2_df,
        ground_truth=gt_df
    )
        
    RESULTS = f'post-review/results/{POST_REVIEW_EXP}/{BLOCKING_TYPE}/{DIR}.csv'
    for responses_df, responses_path, row in tqdm(_get_responses_df(RESULTS, BLOCKING_TYPE, DIR),
                                                  total=102, desc='Responses Remaining'):
        for weight in ['weight','s-weight']:
            tmp_df = responses_df[['D1','D2', 'responses', weight]].copy()
            clust_df = tmp_df[tmp_df['responses']]
            clust_df = clust_df.drop(columns=['responses'])
            edges_with_weights = [(data._ids_mapping_1[str(int(id1))],
                                   data._ids_mapping_2[str(int(id2))],
                                   w) for _, (id1, id2, w) in clust_df.iterrows()]
            g = Graph()
            g.add_weighted_edges_from(edges_with_weights)
            clustering = UniqueMappingClustering()
            clusters = clustering.process(g, data, similarity_threshold=0.0)
            pairs_df : pd.DataFrame = clustering.export_to_df(clusters).astype(int)
            valid_pairs = {tuple(row) for _, row in pairs_df.iterrows()}


            is_valid = tmp_df.set_index(['D1', 'D2']).index.isin(valid_pairs)
            tmp_df['responses'] = is_valid


            cl_reponses_path = responses_path.replace('post-review/responses/',
                                                      'post-review/responses/clustering/')
            tmp_df.to_csv(cl_reponses_path, mode='w+', index=False)
                        
            ev = clustering.evaluate(clusters, verbose=False)
            clustering_df = pd.DataFrame(
            {
                'dataset_1': clean_files[0],
                'dataset_2': clean_files[1],
                'dataset': DIR,
                'model': row['model'],
                'time (sec)': row['time (sec)'] + clustering.execution_time,
                'precision': ev['Precision %'],
                'recall': ev['Recall %'],
                'f1': ev['F1 %'],
                'good_behavior_response_rate': row['good_behavior_response_rate'],
                "examples" : row['examples'],
                'total_matches': len(pairs_df),
                "weights_extracted_from" : wef[BLOCKING_TYPE][weight]
                }, index=[0]
            )
            clustering_path = f'post-review/results/{POST_REVIEW_EXP}/{BLOCKING_TYPE}/{DIR}_clustering.csv'
            HEADER = True if not os.path.exists(clustering_path) else False
            clustering_df.to_csv(clustering_path, index=False, float_format='%.2f', header=HEADER, mode='a+')
