"""Main Experiments Code. Check how well LLMs can do the matching"""
import os
import time
import pandas as pd
import numpy as np
import ollama
from tqdm import tqdm
from inference_utils import (
    _evaluate,
    _load_dataset,
    _convert_to_numpy,
    _create_model,
    _get_response_time,
    _get_good_behavior_response_rate
)
from examples_blocking import examples_dict_list as blocking_examples
from examples import examples_dict_list as original_examples


if __name__ == "__main__":

    BLOCKING_TYPE = "standard_blocking"
    DIR = "D5"

    llms = [
        "gemma3n",
        "qwen2.5",
        "llama3.1",
        "orca2",
        "openhermes",
        "zephyr"
    ]

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

    if BLOCKING_TYPE == "original":
        examples_dict_list = original_examples
    else:
        examples_dict_list = blocking_examples

    for examples, examples_dict in tqdm(examples_dict_list.items(),
                desc="Examples", position=0, leave=True):

        for ll in llms:
            for suffix in ['z', 'ft', 'tf']:
                if suffix == 'z' and examples != "vector_based_examples_dict_1":
                    continue
                for prompt_key in ["p2"]:
                    PROMPT = prompts[prompt_key]
                    # CONFIGURATION: Edit
                    # llm and paths for datasets,
                    # candidate pairs, groundtruth files
                    LLM = f'{ll}-{suffix}-{prompt_key}'
                    # candidate_pairs_dir = "standard_blocking"

                    print(f' ----------- {DIR} {LLM} -------------')

                    dt1_df, dt2_df, cp_df, gt_df, clean_files = _load_dataset(BLOCKING_TYPE, DIR)
                    dt1, dt2, cp, gt_set = _convert_to_numpy(dt1_df, dt2_df, cp_df, gt_df)

                    start = time.time()
                    responses = []
                    num_iterations = len(cp)


                    dt1_ids = {dt1_df.at[i, 'id']: i for i in range(len(dt1_df))}
                    dt2_ids = {dt2_df.at[i, 'id']: i for i in range(len(dt2_df))}

                    PROMPT = _create_model(LLM, ll, PROMPT,
                            suffix,
                            examples_dict, DIR, dt1_df,
                            dt2_df, dt1_ids, dt2_ids)


                    print(f""" --- {LLM} ----\n{PROMPT}\n """)

                    for pair in tqdm(cp,
                            desc="Processing", position=1, leave=False):

                        dt1_id = pair[0]
                        dt2_id = pair[1]


                        r1 = f"Title: {dt1_df.at[dt1_ids[dt1_id], 'title']}, \
                        Associated Name: {dt1_df.at[dt1_ids[dt1_id], 'name']}"
                        r2 = f"Title: {dt2_df.at[dt2_ids[dt2_id], 'title']}, \
                        Associated Name: {dt2_df.at[dt2_ids[dt2_id], 'name']}"

                        query = f"record 1: {r1}, record 2: {r2}. Answer with True. or False."

                        resp = ollama.chat(
                            model = LLM,
                            messages = [{'role': 'user', 'content': query}],
                            options = {'stop': ['\n','.']},
                            stream = False
                        )

                        responses.append(resp['message']['content'])

                        gt_value = (dt1_id, dt2_id) in gt_set

                    end = time.time()

                    cp_df['responses'] =  responses

                    time_seconds =  end - start
                    _get_response_time(start, end)


                    good_behavior_rate = _get_good_behavior_response_rate(responses)

                    cp_df['responses'] = cp_df['responses'].apply(lambda x:
                            'True' in str(x))

                    cp_df.to_csv(f'post-review/responses/{BLOCKING_TYPE}/{DIR}/{DIR}_{LLM}_{examples}.csv',
                                index=False)


                    precision, recall, f1 = _evaluate(gt_set, cp, responses)
                    filtered_cp_df = cp_df[cp_df['responses'] == True]

                    u0, counts0 = np.unique(filtered_cp_df['D1'], return_counts=True)
                    u1, counts1 = np.unique(filtered_cp_df['D2'], return_counts=True)

                    u0 = u0[counts0 == 1]
                    u1 = u1[counts1 == 1]

                    total_pairs = filtered_cp_df.shape[0]

                    non_conflicts = np.isin(filtered_cp_df['D1'], u0) & np.isin(filtered_cp_df['D2'], u1)
                    conflict_pct = (1 - non_conflicts.sum() / total_pairs) * 100 if total_pairs > 0 else 0.0

                    results_df = pd.DataFrame(
                        {
                            'dataset_1': clean_files[0],
                            'dataset_2': clean_files[1],
                            'dataset': DIR,
                            'model': LLM,
                            'time (sec)': time_seconds,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1 ,
                            'good_behavior_response_rate': good_behavior_rate,
                            "examples" : examples,
                            'total_matches': filtered_cp_df.shape[0],
                            "conflicts" : conflict_pct
                        }, index=[0]
                    )
                    ollama.delete(model=LLM)

                    header = not os.path.exists(f'post-review/results/movies/{BLOCKING_TYPE}/{DIR}.csv')
                    results_df.to_csv(f'post-review/results/movies/{BLOCKING_TYPE}/{DIR}.csv', mode='a+',
                        index=False, header=header, float_format='%.2f')
