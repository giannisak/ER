from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time
import ollama
from examples_blocking import examples_dict_list
from ui import ui_fun


llms = [ 
    "gemma3n",  
    "qwen2.5",  
    "llama3.1",  
    "orca2",  
    "openhermes",  
    "zephyr"
]


# examples_dict_list = [vector_based_examples_dict_1, vector_based_examples_dict_2, join_examples_dict_1, join_examples_dict_2]
# examples_list = ['vector_based_examples_dict_1', 'vector_based_examples_dict_2', 'join_examples_dict_1', 'join_examples_dict_2']

prompts = { "p1" : """You are a crowdsourcing worker, working on an entity resolution task.
You will be given two record descriptions and your task is to identify
if the records refer to the same entity or not.
           
You must answer with just one word:
True. if the records are referring to the same entity,
False. if the records are referring to a different entity.""",

"p2" : """You are given two record descriptions and your task is to identify
if the records refer to the same entity or not.

You must answer with just one word:
True. if the records are referring to the same entity,
False. if the records are referring to a different entity."""

}


# dataset = 'D2'
for dataset in ['D2', 'D5', 'D6', 'D7', 'D8' ]:
    
# for dataset in ['D2']:
    cnt = 0
    for examples in examples_dict_list:
        print(f"""
                -----------
                {cnt} / 4
            ---------------
            
            """ )

        examples_dict = examples_dict_list[examples]
        
        print(examples)
        cnt += 1
        # examples = examples_list[i]
        for ll in llms:
            for suffix in ['z', 'ft', 'tf']:
                for prompt_key in ["p2"]:
                    prompt = prompts[prompt_key]
                    # CONFIGURATION: Edit llm and paths for datasets, candidate pairs, groundtruth files
                    llm = f'{ll}-{suffix}-{prompt_key}'
                    candidate_pairs_dir = "standard_blocking"
        
                    if os.path.exists(f'results/{candidate_pairs_dir}/{dataset}.csv'):
                        results_df = pd.read_csv(f'results/{candidate_pairs_dir}/{dataset}.csv')
                        exists = results_df[(results_df['dataset'] == dataset) & 
                                        (results_df['model'] == llm) & (results_df['examples'] == examples)] if suffix != 'z' \
                                else results_df[(results_df['dataset'] == dataset) & (results_df['model'] == llm)]
        
                        if not exists.empty:
                            print(f'{dataset} {llm} DONE')
                            continue
            

                    print(f' ----------- {dataset} {llm} -------------')

                    candidate_pairs = f'data/candidate_pairs/{candidate_pairs_dir}/{dataset}.csv'
                    cp_df_with_rows = pd.read_csv(candidate_pairs)
                    cp_columns = list(cp_df_with_rows.columns)
                    clean_files = [cl.replace("clean", "").replace(".csv", "") for cl in cp_columns]
                    
                    # data_dir = "giannis"
                    dataset_1 = f'data/{dataset}/{clean_files[0]}clean.csv'
                    dataset_2 = f'data/{dataset}/{clean_files[1]}clean.csv'
                    groundtruth = f'data/{dataset}/gtclean.csv'
                    
                    testing = False # Set to True to evaluate the first 100 candidate pairs for testing

                    sep = '|' if dataset!='D3' else '#'

                    # read the files (Edit sep if needed)
                    dt1_df = pd.read_csv(dataset_1, sep=sep)
                    dt2_df = pd.read_csv(dataset_2, sep=sep)
                    
                    gt_df = pd.read_csv(groundtruth, sep=sep)


                    if candidate_pairs_dir == 'original':
                        cp_df = pd.DataFrame({
                            'D1': cp_df_with_rows[cp_columns[0]].map(dt1_df['id']),
                            'D2': cp_df_with_rows[cp_columns[1]].map(dt2_df['id'])
                        })
                    else: 
                        cp_df = cp_df_with_rows
                        cp_df.columns = ['D1','D2']

                    # convert to numpy arrays
                    dt1 = dt1_df.to_numpy()
                    dt2 = dt2_df.to_numpy()
                    cp = cp_df.to_numpy()
                    gt = gt_df.to_numpy()

                    # Convert groundtruth to a set of tuples for O(1) lookup
                    gt_set = set(tuple(row) for row in gt)

                    # cut the indexes
                    dt1 = dt1[:, 1:]
                    dt2 = dt2[:, 1:]


                    # concatenate the strings in each column to a single string, omitting empty elements
                    dt1 = np.array([' '.join([x for x in row if isinstance(x, str)]) for row in dt1])
                    dt2 = np.array([' '.join([x for x in row if isinstance(x, str)]) for row in dt2])

                    # main loop: model iterates through each pair and returns its responses
                    start = time.time()
                    responses = []

                    num_iterations = 100 if testing else len(cp)

                    dt1_ids = {dt1_df.at[i, 'id']: i for i in range(len(dt1_df))}
                    dt2_ids = {dt2_df.at[i, 'id']: i for i in range(len(dt2_df))}


                    #create model 

                    if suffix != 'z': 
                        true_flag = True
                        true_example = []
                        false_example = []
                        for pair in examples_dict[dataset]:
                            r1 = dt1[dt1_ids[pair[0]]]
                            r2 = dt2[dt2_ids[pair[1]]]
                            if true_flag:
                                true_example.append(f"record 1: {r1}\nrecord 2: {r2}\nAnswer: {true_flag}.")
                            else: 
                                false_example.append(f"record 1: {r1}\nrecord 2: {r2}\nAnswer: {true_flag}.")
                            true_flag = False if true_flag else True
                      
                        if suffix == 'ft':
                            example_cnt = 1
                            
                            for example in false_example:
                                prompt = f"""{prompt}\nExample {example_cnt}:\n{example}\n"""
                                example_cnt += 1
                            
                            for example in true_example:
                                prompt = f"""{prompt}\nExample {example_cnt}:\n{example}\n"""
                                example_cnt += 1
                        else: 
                            example_cnt = 1
                            for example in true_example:
                                prompt = f"""{prompt}\nExample {example_cnt}:\n{example}\n"""
                                example_cnt += 1

                            for example in false_example:
                                prompt = f"""{prompt}\nExample {example_cnt}:\n{example}\n"""
                                example_cnt += 1
                            
                            
                    print(f""" --- {llm} ----\n{prompt}\n """)
                    ollama.create(model=llm, from_=ll, system=prompt)

                    for i in tqdm(range(num_iterations), desc="Processing"):
                        
                        dt1_id = cp[i][0]
                        dt2_id = cp[i][1]

                        r1 = dt1[dt1_ids[dt1_id]]
                        r2 = dt2[dt2_ids[dt2_id]]

                        query = f"record 1: {r1}, record 2: {r2}. Answer with True. or False."

                        resp = ollama.chat(
                            model = llm,
                            messages = [{'role': 'user', 'content': query}],
                            options = {'stop': ['\n','.']},
                            stream = False
                        )
                        
                        responses.append(resp['message']['content'])

                        gt_value = 'True' if (dt1_id, dt2_id) in gt_set else 'False'

                    # print(f"groundtruth value: {gt_value}")
                    # print(f"pair: {[dt1_index, dt2_index]}")
                    # print(" ")

                    end = time.time()

                    cp_df['responses'] =  responses
                    cp_df.to_csv(f'responses/{candidate_pairs_dir}/{dataset}/{dataset}_{llm}_{examples}.csv', index=False)

                    # model's response time
                    time_seconds = end - start  
                    hours, remainder = divmod(time_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    print("Response Time: {:02}:{:02}:{:.2f}".format(int(hours), int(minutes), seconds))

                    # model's 'good behavior' response rate
                    good_responses = sum(response == 'True' or response == 'False' for response in responses)
                    good_behavior_rate = good_responses / len(responses)
                    print("Good Behavior Response Rate:", good_behavior_rate)

                # #model's conflict rate
                # record = cp[0][1]
                # count_true = 0
                # conflicts = 0
                # conflict_records = 0

                # for i in range(num_iterations):
                #     if record == cp[i][1]:
                #         if responses[i] == 'True':
                #             count_true += 1
                #     else:
                #         if count_true > 1:
                #             conflict_records += 1
                #             conflicts += count_true - 1
                #         record = cp[i][1]
                #         count_true = 0
                #         if responses[i] == 'True':
                #             count_true += 1

                # if count_true > 1:
                #     conflict_records += 1
                #     conflicts += count_true - 1

                # print("Conflicts:", conflicts)
                # print("Conflicted Records:", conflict_records)
                # conflict_rate = conflict_records / (cp[i][1]+1)
                # print("Conflict Rate per Record:", conflict_rate)

                    #evaluation metrics
                    true_labels = [1 if tuple(pair) in gt_set else 0 for pair in cp[:num_iterations]]
                    predicted_labels = [1 if resp == 'True' else 0 for resp in responses]

                    conf_matrix = confusion_matrix(true_labels, predicted_labels)

                    TP = conf_matrix[1, 1]
                    FP = conf_matrix[0, 1]
                    TN = conf_matrix[0, 0]
                    FN = conf_matrix[1, 0]

                    # accuracy = accuracy_score(true_labels, predicted_labels)
                    precision = precision_score(true_labels, predicted_labels)
                    recall = recall_score(true_labels, predicted_labels)
                    f1 = f1_score(true_labels, predicted_labels)

                    # print("\nTrue Positives:", TP)
                    # print("False Positives:", FP)
                    # print("True Negatives:", TN)
                    # print("False Negatives:", FN)

                    # print("\nAccuracy:", accuracy)
                    print("Precision:", precision)
                    print("Recall:", recall)
                    print("F1 Score:", f1)
                    print(" ")

                    # save evaluation summary to same folder as datasets
                    dataset_dir = '/'.join(dataset_1.split('/')[:-1])
                    summary_filename = f"{dataset_dir}/{llm}_results_{examples}.txt"

                    with open(summary_filename, 'w') as f:
                        f.write("Response Time: {:02}h:{:02}m:{:.2f}s\n".format(int(hours), int(minutes), seconds))
                        f.write(f"Good Behavior Response Rate: {good_behavior_rate:.2f}\n")
                        f.write(f"Precision: {precision:.3f}\n")
                        f.write(f"Recall: {recall:.3f}\n")
                        f.write(f"F1 Score: {f1:.3f}\n")

                    
                    results_df = pd.DataFrame(
                        {
                            'dataset_1': clean_files[0],
                            'dataset_2': clean_files[1],
                            'dataset': dataset,
                            'model': llm,
                            'time (sec)': time_seconds,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'good_behavior_response_rate': good_behavior_rate,
                            "examples" : examples
                        }, index=[0]
                    )
                    ollama.delete(model=llm)
                    
                    if os.path.exists(f'results/{candidate_pairs_dir}/{dataset}.csv'):
                        results_df.to_csv(f'results/{candidate_pairs_dir}/{dataset}.csv', mode='a+', index=False, header=False)
                    else:
                        results_df.to_csv(f'results/{candidate_pairs_dir}/{dataset}.csv', mode='a+', index=False, header=True)
                        
                    print(f"Summary saved to {summary_filename}")

                    # model's responses
                    for i in range(len(responses)):
                        if responses[i] != 'True' and responses[i] != 'False':
                            responses[i] = 'False' 

                    if 'ft' in llm or 'tf' in llm:
                        responses_filename = f"{dataset_dir}/{llm}_responses.txt"
                        with open(responses_filename, 'w') as file:
                            for response in responses:
                                file.write(response + '\n')
                        print(f"Responses saved to {responses_filename} for union/intersection.")
    ui_fun(dataset, candidate_pairs_dir)