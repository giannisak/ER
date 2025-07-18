import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time
import os


# Evaluation function
def evaluate(predictions, label):
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

    # Save summary to file
    result_filename = f"{dataset_dir}/zephyr_{label}_results.txt"
    with open(result_filename, 'w') as f:
        f.write(f"Precision: {precision:.3f}\n")
        f.write(f"Recall: {recall:.3f}\n")
        f.write(f"F1 Score: {f1:.3f}\n")
    print(f"Summary saved to {result_filename}")

    return precision, recall, f1



llms = [ 
    "gemma3n",  
    "qwen2.5",  
    "llama3.1",  
    "orca2",  
    "openhermes",  
    "zephyr"
]
for dataset in ['D2', 'D5', 'D6', 'D7', 'D8' ]:
    for ll in llms:



        results_df = pd.read_csv("results.csv")

        

        # CONFIGURATION: Edit the two model names and dataset files
        model1 = f'{ll}-ft'
        
        row1 = results_df[(results_df['dataset'] == dataset) 
                          & (results_df['model'] == model1)].iloc[0]
        

        model2 = f'{ll}-tf'
        row2 = results_df[(results_df['dataset'] == dataset) 
                          & (results_df['model'] == model2)].iloc[0]
        

        candidate_pairs = f'candidate_pairs/{dataset}.csv'
        groundtruth = f'data_clean/{dataset}/gtclean.csv'
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

        # Load model responses
        with open(f"{dataset_dir}/{model1}_responses.txt", 'r') as f1:
            responses1 = [line.strip() for line in f1.readlines()]

        with open(f"{dataset_dir}/{model2}_responses.txt", 'r') as f2:
            responses2 = [line.strip() for line in f2.readlines()]

        start = time.time()

        # Compute union and intersection of responses
        union = ['True' if r1 == 'True' or r2 == 'True' else 'False' for r1, r2 in zip(responses1, responses2)]
        intersection = ['True' if r1 == 'True' and r2 == 'True' else 'False' for r1, r2 in zip(responses1, responses2)]

        end = time.time()
        time_seconds = end - start  

        # Run evaluation


        precision, recall, f1 = evaluate(union, 'union')

        union_sym = "U"
        intersection_sym = "∩"
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
                'good_behavior_response_rate': (row1['good_behavior_response_rate'] + row2['good_behavior_response_rate']) / 2
            }, index=[0]
        )
        if os.path.exists('results.csv'):
            new_results_df.to_csv('results.csv', mode='a+', index=False, header=False)
        else:
            new_results_df.to_csv('results.csv', mode='a+', index=False, header=True)
                

        precision, recall, f1 = evaluate(intersection, 'intersection')
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
                'good_behavior_response_rate': (row1['good_behavior_response_rate'] + row2['good_behavior_response_rate']) / 2
            }, index=[0]
        )
        if os.path.exists('results.csv'):
            new_results_df.to_csv('results.csv', mode='a+', index=False, header=False)
        else:
            new_results_df.to_csv('results.csv', mode='a+', index=False, header=True)
                