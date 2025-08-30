from pyjedai.clustering import UniqueMappingClustering
import pandas as pd
from pyjedai.datamodel import Data
import os
from networkx import Graph
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


if  __name__ == '__main__': 
    datasets = ["D2", "D5", "D6", "D7", "D8"]
    for dataset in datasets:
        cp = pd.read_csv(f'data/candidate_pairs/original/{dataset}.csv')
        cp_columns = list(cp.columns)
        clean_files = [cl.replace("clean", "").replace(".csv", "") for cl in cp_columns]
        
        sep = '|' if dataset!='D3' else '#'

        d1 = pd.read_csv(f"data/{dataset}/{clean_files[0]}clean.csv", sep=sep)
        d2 = pd.read_csv(f"data/{dataset}/{clean_files[1]}clean.csv", sep=sep)
        gt = pd.read_csv(f"data/{dataset}/gtclean.csv", sep=sep)
        gt = gt.to_numpy()

        # Convert groundtruth to a set of tuples for O(1) lookup
        gt_set = set(tuple(row) for row in gt)


                
        repsonses_set = set()
        for candidate_pairs in ['original', 'standard_blocking']: 
            results_df = pd.read_csv(f'results/{candidate_pairs}/{dataset}.csv')
            for index, row in results_df.iterrows():
                llm : str = row['model']
                original_llm_name = llm
                if "U" in llm or "∩" in llm:
                    continue 
                
                    # llm = llm.replace(" U ", "_union_") if "U" in llm else llm.replace(" ∩ ", "_intersection_") 

                examples = row['examples']
                responses_path = f'responses/{candidate_pairs}/{dataset}/{dataset}_{llm}_{examples}.csv'
                    
                if '-z' in llm and not os.path.exists(responses_path):
                    responses_path = f'responses/{candidate_pairs}/{dataset}/{dataset}_{llm}.csv'
                        
                responses_df = pd.read_csv(responses_path)
                responses_df['responses'] = responses_df['responses'].astype(str)
                cp = responses_df[['D1','D2']]
                cp = cp.to_numpy()

                
                
                true_labels = [1 if tuple(pair) in gt_set else 0 for pair in cp]
                predicted_labels = responses_df['responses'].apply(lambda x: 1 if 'true' in x.lower() else 0).to_list() 

                conf_matrix = confusion_matrix(true_labels, predicted_labels)

                TP = conf_matrix[1, 1]
                FP = conf_matrix[0, 1]
                TN = conf_matrix[0, 0]
                FN = conf_matrix[1, 0]

                # accuracy = accuracy_score(true_labels, predicted_labels)
                precision = precision_score(true_labels, predicted_labels)
                recall = recall_score(true_labels, predicted_labels)
                f1 = f1_score(true_labels, predicted_labels)
                my_dict  = {
                    "precision" : precision,
                    "recall" : recall,
                    "f1" : f1
                }
                columns = ['precision', 'recall', 'f1']
                
                for col in columns:
                    if row[col] != my_dict[col]: 
                        results_df.at[index, col] = my_dict[col]
                        print(f'{col} -- prev : {row[col]}  -- now: {my_dict[col]} ')
            
            results_df.to_csv(f'results/{candidate_pairs}/{dataset}.csv', index=False, mode='w+')                

                
                    
                     
                    
                    

  
        #             repsonses_set.update(responses_df['responses'].astype(str).unique())
                    
                    
                    
        # print(repsonses_set)

        # sorted_strings = sorted(repsonses_set, key=len)

        # with open('responses/reponses.txt', 'w+') as w:
        #     for item in sorted_strings:
        #         w.write(f'{item}\n')
            
            
                    # for weight in ['weight','s-weight']:
                    #     tmp_df = responses_df[['D1','D2', 'responses', weight]]
                    #     tmp_df['responses'] = tmp_df['responses'].astype(str)

                    #     tmp_df = tmp_df[tmp_df['responses'] == 'True']
                        
                    #     tmp_df.drop(columns=['responses'], inplace=True)

                    #     edges_with_weights = [(data._ids_mapping_1[str(int(id1))], data._ids_mapping_2[str(int(id2))], w) for _, (id1, id2, w) in tmp_df.iterrows()]
                    #     g = Graph()
                    #     g.add_weighted_edges_from(edges_with_weights)
                    #     clustering = UniqueMappingClustering()
                    #     clusters = clustering.process(g, data, similarity_threshold=0.0)
                    #     pairs_df = clustering.export_to_df(clusters)
                        
                    #     ev = clustering.evaluate(clusters)
                    #     clustering_df = pd.DataFrame(
                    #     {
                    #         'dataset_1': clean_files[0],
                    #         'dataset_2': clean_files[1],
                    #         'dataset': dataset,
                    #         'model': original_llm_name,
                    #         'time (sec)': row['time (sec)'] + clustering.execution_time,
                    #         'precision': ev['Precision %'],
                    #         'recall': ev['Recall %'],
                    #         'f1': ev['F1 %'],
                    #         'good_behavior_response_rate': row['good_behavior_response_rate'],
                    #         "examples" : examples,
                    #         'total_matches': len(pairs_df),
                    #         "weights_extracted_from" : wef[candidate_pairs][weight]
                    #     }, index=[0]
                    #     )
                    #     clustering_path = f'results/{candidate_pairs}/{dataset}_clustering.csv'
                    #     HEADER = True if not os.path.exists(clustering_path) else False
                    #     clustering_df.to_csv(clustering_path, index=False, header=HEADER, mode='a+')
                        