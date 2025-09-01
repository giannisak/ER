from pyjedai.clustering import UniqueMappingClustering
import pandas as pd
from pyjedai.datamodel import Data
import os
from networkx import Graph

if  __name__ == '__main__': 
    datasets = ["D2", "D5", "D6", "D7", "D8"]
    # datasets = ["D2" , "D7"]
    
    for dataset in datasets:
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
        
        wef = {'original' : {'weight' : 'TopKJoin' ,  's-weight' : 'distilroberta'},
        'standard_blocking' : {'weight' : 'MetaBlocking-Method' ,  's-weight' : 'distilroberta'},
        }
                
        for candidate_pairs in ['original', 'standard_blocking']: 
        # for candidate_pairs in ['standard_blocking']:
            
            results_df = pd.read_csv(f'results/{candidate_pairs}/{dataset}.csv')
            for _, row in results_df.iterrows():

                    llm : str = row['model']
                    original_llm_name = llm
                    if  "U" in llm or "∩" in llm: 
                        continue
                        llm = llm.replace(" U ", "_union_") if "U" in llm else llm.replace(" ∩ ", "_intersection_") 

                    examples = row['examples']
                    responses_path = f'responses/{candidate_pairs}/{dataset}/{dataset}_{llm}_{examples}.csv'
                        
                    if '-z' in llm and not os.path.exists(responses_path):
                        responses_path = f'responses/{candidate_pairs}/{dataset}/{dataset}_{llm}.csv'
                            
                    responses_df = pd.read_csv(responses_path)
                    print(len(responses_df))
                    for weight in ['weight','s-weight']:
                        tmp_df = responses_df[['D1','D2', 'responses', weight]]
                        print(len(tmp_df))
                        
                        tmp_df['responses'] = tmp_df['responses'].astype(str)
                        tmp_df['responses'].apply(lambda x: 'True' if 'true' in x.lower() else  'False')
                        
                        

                        clust_df = tmp_df[tmp_df['responses'] == 'True']
                        
                        clust_df = clust_df.drop(columns=['responses'])

                        edges_with_weights = [(data._ids_mapping_1[str(int(id1))], data._ids_mapping_2[str(int(id2))], w) for _, (id1, id2, w) in clust_df.iterrows()]
                        g = Graph()
                        g.add_weighted_edges_from(edges_with_weights)
                        clustering = UniqueMappingClustering()
                        clusters = clustering.process(g, data, similarity_threshold=0.0)
                        pairs_df : pd.DataFrame = clustering.export_to_df(clusters).astype(int)
                        
                        # print(pairs_df)
                        valid_pairs = set(tuple(row) for _,row in pairs_df.iterrows())
                        # print(valid_pairs)
                        tmp_df['responses'] = "False"
                       

                        mask = (tmp_df['responses'] == 'False') & (tmp_df[['D1', 'D2']].apply(tuple, axis=1).isin(valid_pairs))
                        tmp_df.loc[mask, 'responses'] = 'True'

                        cl_reponses_path = f'responses/clustering/{candidate_pairs}/{dataset}/{dataset}_{llm}_{examples}_{weight}.csv'

                        if '-z' in llm and not os.path.exists(responses_path):
                            cl_reponses_path = f'responses/clustering/{candidate_pairs}/{dataset}/{dataset}_{llm}_{weight}.csv'
  
                        tmp_df.to_csv(cl_reponses_path, mode='w+', index=False)
                        
                        ev = clustering.evaluate(clusters)
                        clustering_df = pd.DataFrame(
                        {
                            'dataset_1': clean_files[0],
                            'dataset_2': clean_files[1],
                            'dataset': dataset,
                            'model': original_llm_name,
                            'time (sec)': row['time (sec)'] + clustering.execution_time,
                            'precision': ev['Precision %'],
                            'recall': ev['Recall %'],
                            'f1': ev['F1 %'],
                            'good_behavior_response_rate': row['good_behavior_response_rate'],
                            "examples" : examples,
                            'total_matches': len(pairs_df),
                            "weights_extracted_from" : wef[candidate_pairs][weight]
                        }, index=[0]
                        )
                        clustering_path = f'results/{candidate_pairs}/{dataset}_clustering.csv'
                        HEADER = True if not os.path.exists(clustering_path) else False
                        # clustering_df.to_csv(clustering_path, index=False, header=HEADER, mode='a+')
                        