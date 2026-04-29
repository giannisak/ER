
from pyjedai.clustering import ConnectedComponentsClustering
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding
import pandas as pd
from pyjedai.datamodel import Data
import os
from tqdm.auto import  tqdm
from networkx import Graph
from gpt_utils import _load_dataset_cora, _get_responses_df

if  __name__ == '__main__':


    BLOCKING_TYPE = 'standard_blocking'
    DIR = 'cora'
    POST_REVIEW_EXP = "cora"
    dt1_df, cp_df, gt_df  = _load_dataset_cora(BLOCKING_TYPE, DIR)
    data = Data(
        dataset_1=dt1_df,
        id_column_name_1='Entity Id',
        ground_truth=gt_df
    )
        
    RESULTS = f'post-review/results/{POST_REVIEW_EXP}/{BLOCKING_TYPE}/{DIR}.csv'
    for responses_df, responses_path, row in tqdm(_get_responses_df(RESULTS, BLOCKING_TYPE, DIR),
                                                  total=102, desc='Responses Remaining'):
        clust_df = responses_df[responses_df['responses']]
        df_edges = set(frozenset([u, v]) for u, v in zip(clust_df['D1'], clust_df['D2']))
        emb = EmbeddingsNNBlockBuilding(vectorizer='sdistilroberta', similarity_search='faiss')

        blocks, g = emb.build_blocks(data, top_k=5, with_entity_matching=True, load_embeddings_if_exist=True)
        edges_to_keep = [(u, v) for u, v in g.edges() if frozenset([u, v]) in df_edges]
        subgraph = g.edge_subgraph(edges_to_keep).copy()
        clustering = ConnectedComponentsClustering()
        clusters = clustering.process(subgraph, data, similarity_threshold=0.0)
        pairs_df : pd.DataFrame = clustering.export_to_df(clusters).astype(int)
        valid_pairs = {tuple(row) for _, row in pairs_df.iterrows()}


        is_valid = clust_df.set_index(['D1', 'D2']).index.isin(valid_pairs)
        clust_df['responses'] = is_valid


        cl_reponses_path = responses_path.replace('post-review/responses/',
                                                  'post-review/responses/clustering/')
        clust_df.to_csv(cl_reponses_path, mode='w+', index=False)
                        
        ev = clustering.evaluate(clusters, verbose=False)
        clustering_df = pd.DataFrame(
        {
            'dataset_1': 'cora',
            'dataset': 'cora',
            'model': row['model'],
            'time (sec)': row['time (sec)'] + clustering.execution_time,
            'precision': ev['Precision %'],
            'recall': ev['Recall %'],
            'f1': ev['F1 %'],
            'good_behavior_response_rate': row['good_behavior_response_rate'],
            "examples" : row['examples'],
            'total_matches': len(pairs_df),
            "weights_extracted_from" : 's-weight'
            }, index=[0]
        )
        clustering_path = f'post-review/results/{POST_REVIEW_EXP}/{BLOCKING_TYPE}/{DIR}_clustering.csv'
        HEADER = True if not os.path.exists(clustering_path) else False
        clustering_df.to_csv(clustering_path, index=False, float_format='%.2f', header=HEADER, mode='a+')
