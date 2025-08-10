import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time
import os
from examples import examples_dict_list


# Evaluation function
def evaluate(predictions, label, gt_set, cp):
    true_labels = [1 if tuple(pair) in gt_set else 0 for pair in cp]
    predicted_labels = [1 if resp == 'True' else 0 for resp in predictions]

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    TP = conf_matrix[1, 1]
    FP = conf_matrix[0, 1]
    TN = conf_matrix[0, 0]
    FN = conf_matrix[1, 0]

    # accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f"{label.upper()} Evaluation")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print(" ")

    # # Save summary to file
    # result_filename = f"{dataset_dir}/zephyr_{label}_results.txt"
    # with open(result_filename, 'w') as f:
    #     f.write(f"Precision: {precision:.3f}\n")
    #     f.write(f"Recall: {recall:.3f}\n")
    #     f.write(f"F1 Score: {f1:.3f}\n")
    # print(f"Summary saved to {result_filename}")

    return precision, recall, f1



llms = [ 
    "gemma3n",  
    "qwen2.5",  
    "llama3.1",  
    "orca2",  
    "openhermes",  
    "zephyr"
]

union_sym = "U"
intersection_sym = "∩"


# for dataset in ['D2', 'D5', 'D6', 'D7', 'D8' ]:
def ui_fun(dataset, candidate_pairs_dir):
# for dataset in ['D8']:
    for ll in llms:
        for prompt in ['p2']:
            for examples in examples_dict_list:
                results_filename = f'results/{candidate_pairs_dir}/{dataset}.csv'
                

                results_df = pd.read_csv(results_filename)

        

                # CONFIGURATION: Edit the two model names and dataset files
                model1 = f'{ll}-ft-{prompt}'
                
                row1 = results_df[(results_df['dataset'] == dataset) 
                                & (results_df['model'] == model1)
                                & (results_df['examples'] == examples)].iloc[0]
                

                model2 = f'{ll}-tf-{prompt}'
                row2 = results_df[(results_df['dataset'] == dataset) 
                                & (results_df['model'] == model2)
                                & (results_df['examples'] == examples)].iloc[0]
                
                union_exists = results_df[(results_df['dataset'] == dataset) & 
                        (results_df['model'] == f"{model1} {union_sym} {model2}") & 
                        (results_df['examples'] == examples)] 
                intersection_exists = results_df[(results_df['dataset'] == dataset) & 
                        (results_df['model'] == f"{model1} {intersection_sym} {model2}") & 
                        (results_df['examples'] == examples)] 
                
                    
                if not union_exists.empty and not intersection_exists.empty:
                    print(f'{dataset} {model1} {model2} {examples} union and intersection DONE')
                    continue


                candidate_pairs = f'data/candidate_pairs/{candidate_pairs_dir}/{dataset}.csv'
                groundtruth = f'data/{dataset}/gtclean.csv'
                cp_df = pd.read_csv(candidate_pairs)
                cp_columns = list(cp_df.columns)
                clean_files = [cl.replace("clean", "").replace(".csv", "") for cl in cp_columns]

                # Load data
                cp = pd.read_csv(candidate_pairs).to_numpy()
                # cp = cp[:100]  # Testing mode – only first 100 pairs
                gt = pd.read_csv(groundtruth, sep='|').to_numpy()
                gt_set = set(tuple(row) for row in gt)

                # Get dataset directory
                dataset_dir = f'data_clean/{dataset}'
                
                responses1_df = pd.read_csv(f'responses/{candidate_pairs_dir}/{dataset}/{dataset}_{model1}_{examples}.csv')
                responses2_df = pd.read_csv(f'responses/{candidate_pairs_dir}/{dataset}/{dataset}_{model2}_{examples}.csv')


                responses1 = responses1_df['responses'].astype('str').tolist()
                responses2 = responses2_df['responses'].astype('str').tolist()
                
                # Load model responses
                # with open(f"{dataset_dir}/{model1}_responses_{examples}.txt", 'r') as f1:
                #     responses1 = [line.strip() for line in f1.readlines()]

                # with open(f"{dataset_dir}/{model2}_responses_{examples}.txt", 'r') as f2:
                #     responses2 = [line.strip() for line in f2.readlines()]

                start = time.time()

                # Compute union and intersection of responses
                union = ['True' if r1 == 'True' or r2 == 'True' else 'False' for r1, r2 in zip(responses1, responses2)]
                intersection = ['True' if r1 == 'True' and r2 == 'True' else 'False' for r1, r2 in zip(responses1, responses2)]

                end = time.time()
                time_seconds = end - start  

                # Run evaluation


                responses1_df['responses'] = union
                responses2_df['responses'] = intersection
                
                responses1_df.to_csv(f'responses/{candidate_pairs_dir}/{dataset}/{dataset}_{model1}_union_{model2}_{examples}.csv')
                responses2_df.to_csv(f'responses/{candidate_pairs_dir}/{dataset}/{dataset}_{model1}_intersection_{model2}_{examples}.csv')

                
                if isinstance(responses1_df.iloc[0]['responses'], str):    
                    filtered_cp_df = responses1_df[responses1_df['responses'] == 'True']
                else: 
                    filtered_cp_df = responses1_df[responses1_df['responses'] == True]


                d1_list = filtered_cp_df['D1'].to_list()
                d2_list = filtered_cp_df['D2'].to_list()
                d1_set = set(d1_list)
                d2_set = set(d2_list)
                


                precision, recall, f1 = evaluate(union, 'union', gt_set, cp)

                if union_exists.empty:
                    new_results_df = pd.DataFrame(
                        {
                            'dataset_1': clean_files[0],
                            'dataset_2': clean_files[1],
                            'dataset': dataset,
                            'model': f"{model1} {union_sym} {model2}",
                            'time (sec)': time_seconds + row1['time (sec)'] + row2['time (sec)'],
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'good_behavior_response_rate': (row1['good_behavior_response_rate'] + row2['good_behavior_response_rate']) / 2,
                            'examples' : examples, 
                            'total_matches': len(d1_list),
                            "D1_conflicts" : (len(d1_list) - len(d1_set))/len(d1_list),
                            "D2_conflicts" : (len(d2_list) - len(d2_set))/len(d1_list),
                            
                        }, index=[0]
                    )
                    if os.path.exists(results_filename):
                        new_results_df.to_csv(results_filename, mode='a+', index=False, header=False)
                    else:
                        new_results_df.to_csv(results_filename, mode='a+', index=False, header=True)
                        

 
                if isinstance(responses2_df.iloc[0]['responses'], str):    
                    filtered_cp_df = responses2_df[responses2_df['responses'] == 'True']
                else: 
                    filtered_cp_df = responses2_df[responses2_df['responses'] == True]


                d1_list = filtered_cp_df['D1'].to_list()
                d2_list = filtered_cp_df['D2'].to_list()
                d1_set = set(d1_list)
                d2_set = set(d2_list)
                
                precision, recall, f1 = evaluate(intersection, 'intersection', gt_set, cp)
                if intersection_exists.empty:
                    new_results_df = pd.DataFrame(
                        {
                            'dataset_1': clean_files[0],
                            'dataset_2': clean_files[1],
                            'dataset': dataset,
                            'model': f"{model1} {intersection_sym} {model2}",
                            'time (sec)': time_seconds + row1['time (sec)'] + row2['time (sec)'],
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'good_behavior_response_rate': (row1['good_behavior_response_rate'] + row2['good_behavior_response_rate']) / 2,
                            'examples' : examples,
                            'total_matches': len(d1_list),
                            "D1_conflicts" : (len(d1_list) - len(d1_set))/len(d1_list),
                            "D2_conflicts" : (len(d2_list) - len(d2_set))/len(d1_list),
                        }, index=[0]
                    )
                    if os.path.exists(results_filename):
                        new_results_df.to_csv(results_filename, mode='a+', index=False, header=False)
                    else:
                        new_results_df.to_csv(results_filename, mode='a+', index=False, header=True)
                            