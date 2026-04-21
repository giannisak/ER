import time
import numpy as np
import pandas as pd
from gpt_utils import _load_dataset, _convert_to_numpy, _evaluate
from examples_blocking import examples_dict_list as blocking_examples
from examples import examples_dict_list as original_examples


if  __name__ == '__main__':

    BLOCKING_TYPE = "standard_blocking"
    DIR = "D8"
    LLMs = [
        "gpt-5.4",
    ]

    RESULTS = f'post-review/results/gpt/{BLOCKING_TYPE}'
    PROMPT = 'p2'

    UNION_SYM = "U"
    INTERSECTION_SYM = "∩"

    WEIGHTS_DICT =  {
        'original' : {'weight' : 'TopKJoin',
                      's-weight' : 'distilroberta'},
        'standard_blocking' : {'weight' : 'MetaBlocking-Method',
                               's-weight' : 'distilroberta'},
        }

    RESPONSES = f"post-review/responses/{BLOCKING_TYPE}/{DIR}"
    if BLOCKING_TYPE == "original":
        examples_dict_list = original_examples
    else:
        examples_dict_list = blocking_examples

    for ll in LLMs:
        for examples in examples_dict_list:
            results_filename = f'{RESULTS}/{DIR}.csv'
            results_df = pd.read_csv(results_filename, sep= ',')
            # CONFIGURATION: Edit the two model names and dataset files
            model1 = f'{ll}-ft-{PROMPT}'

            row1 = results_df[(results_df['dataset'] == DIR)
                            & (results_df['model'] == model1)
                            & (results_df['examples'] == examples)].iloc[0]
                

            model2 = f'{ll}-tf-{PROMPT}'
            row2 = results_df[(results_df['dataset'] == DIR)
                            & (results_df['model'] == model2)
                            & (results_df['examples'] == examples)].iloc[0]

            dt1_df, dt2_df, cp_df, gt_df, clean_files = _load_dataset(BLOCKING_TYPE, DIR)
            dt_1, dt_2, cp, gt_set = _convert_to_numpy(dt1_df, dt2_df, cp_df, gt_df)


            responses1_df = pd.read_csv(f'{RESPONSES}/{DIR}_{model1}_{examples}.csv')
            responses2_df = pd.read_csv(f'{RESPONSES}/{DIR}_{model2}_{examples}.csv')
            start = time.time()

            union = responses1_df['responses'].values | responses2_df['responses'].values
            intersection = responses1_df['responses'].values & responses2_df['responses'].values

            print(union)

            end = time.time()
            time_seconds = end - start
            time_col = 'time (sec)'
            final_time = time_seconds + row1[time_col] + row2[time_col]

            responses1_df['responses'] = union
            responses2_df['responses'] = intersection
                    
            responses1_df.to_csv(f'{RESPONSES}/{DIR}_{model1}_union_{model2}_{examples}.csv')
            responses2_df.to_csv(f'{RESPONSES}/{DIR}_{model1}_intersection_{model2}_{examples}.csv')


            filtered_cp_df = responses1_df[responses1_df['responses'] == True]


            d1_list = filtered_cp_df['D1'].to_list()
            d2_list = filtered_cp_df['D2'].to_list()
            d1_set = set(d1_list)
            d2_set = set(d2_list)

            u0, counts0 = np.unique(filtered_cp_df['D1'], return_counts=True)
            u1, counts1 = np.unique(filtered_cp_df['D2'], return_counts=True)

            u0 = u0[counts0 == 1]
            u1 = u1[counts1 == 1]

            total_pairs = filtered_cp_df.shape[0]

            non_conflicts = np.isin(filtered_cp_df['D1'], u0) & np.isin(filtered_cp_df['D2'], u1)
            conflict_pct = (1 - non_conflicts.sum() / total_pairs) * 100 if total_pairs > 0 else 0.0

            precision, recall, f1 = _evaluate(gt_set, cp, union.tolist())
            new_results_df = pd.DataFrame(
        {
                'dataset_1': clean_files[0],
                'dataset_2': clean_files[1],
                'dataset': DIR,
                'model': f"{model1} {UNION_SYM} {model2}",
                time_col: final_time,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'good_behavior_response_rate': (row1['good_behavior_response_rate'] + row2['good_behavior_response_rate']) / 2,
                "examples" : examples,
                'total_matches': total_pairs,
                'conflicts': conflict_pct
            }, index=[0]
            )

            new_results_df.to_csv(results_filename, mode='a+',
                              index=False, header=False,  float_format='%.2f')



            filtered_cp_df = responses2_df[responses2_df['responses'] == True]

            d1_list = filtered_cp_df['D1'].to_list()
            d2_list = filtered_cp_df['D2'].to_list()
            d1_set = set(d1_list)
            d2_set = set(d2_list)

            u0, counts0 = np.unique(filtered_cp_df['D1'], return_counts=True)
            u1, counts1 = np.unique(filtered_cp_df['D2'], return_counts=True)

            u0 = u0[counts0 == 1]
            u1 = u1[counts1 == 1]

            total_pairs = filtered_cp_df.shape[0]

            non_conflicts = np.isin(filtered_cp_df['D1'], u0) & np.isin(filtered_cp_df['D2'], u1)
            conflict_pct = (1 - non_conflicts.sum() / total_pairs) * 100 if total_pairs > 0 else 0.0

            precision, recall, f1 = _evaluate(gt_set, cp, intersection.tolist())

            new_results_df = pd.DataFrame(
            {
                    'dataset_1': clean_files[0],
                    'dataset_2': clean_files[1],
                    'dataset': DIR,
                    'model': f"{model1} {INTERSECTION_SYM} {model2}",
                    time_col: final_time,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'good_behavior_response_rate': (row1['good_behavior_response_rate'] + row2['good_behavior_response_rate']) / 2,
                    "examples" : examples,
                    'total_matches': total_pairs,
                    'conflicts': conflict_pct
                }, index=[0]
            )
            new_results_df.to_csv(results_filename, mode='a+',
                              index=False, header=False, float_format='%.2f')
