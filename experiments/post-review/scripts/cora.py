import time
import os
from tqdm.auto import tqdm
import ollama
import pandas as pd
import numpy as np
from examples import examples_dict_list as original_examples
from gpt_utils import (
    _load_dataset_cora,
    _convert_to_numpy_cora,
    _create_model_cora,
    _get_good_behavior_response_rate,
    _evaluate
)
if __name__ == '__main__':
    BLOCKING_TYPE = "original"
    DIR = "50K"

    POST_REVIEW_EXP = "scalability"
    llms = [
        # "gemma3n",
        "qwen2.5",
        # "llama3.1",
        # "orca2",
        # "openhermes",
        # "zephyr"
    ]

    RESULTS = f'post-review/results/{POST_REVIEW_EXP}/{BLOCKING_TYPE}/{DIR}.csv'
    RESPONSES = f"post-review/responses/{BLOCKING_TYPE}/{DIR}"

    prompts = {
        "p2" : (
            "You are given two record descriptions and your task is to identify "
            "if the records refer to the same entity or not. "
            "You must answer with just one word: "
            "True. if the records are referring to the same entity, "
            "False. if the records are referring to a different entity."
        )
    }
    examples_dict_list = original_examples
    for examples, examples_dict in tqdm(examples_dict_list.items(),
                                        desc="Examples", position=0, leave=True):
        for ll in llms:
            for suffix in ['z', 'ft', 'tf']:
                if suffix == 'z' and \
                        examples != "vector_based_examples_dict_1":
                    continue
                for prompt_key in ["p2"]:

                    PROMPT = prompts[prompt_key]
                    LLM = f'{ll}-{suffix}-{prompt_key}'
                    print(f' ----------- {DIR} {LLM} -------------')
                    responses_filename = f'post-review/responses/{BLOCKING_TYPE}/{DIR}/{DIR}_{LLM}_{examples}.csv'
                    if os.path.exists(responses_filename):
                        continue
                    dt1_df, cp_df, gt_df = _load_dataset_cora(BLOCKING_TYPE, DIR)
                    dt_1, cp, gt_set = _convert_to_numpy_cora(dt1_df, cp_df, gt_df)

                    start = time.time()
                    responses = []
                    num_iterations = len(cp)

                    dt1_ids = {int(dt1_df.at[i, 'Id']): i for i in range(len(dt1_df))}
                    PROMPT = _create_model_cora(LLM, ll, PROMPT,
                                           suffix,
                                           examples_dict, DIR, dt_1, dt1_ids)


                    print(f""" --- {LLM} ----\n{PROMPT}\n """)
                    all_messages = []
                    for pair in tqdm(cp,
                                     desc="Processing", position=1, leave=False):
                        dt1_id = pair[0]
                        dt2_id = pair[1]

                        r1 = dt_1[dt1_ids[pair[0]]]
                        r2 = dt_1[dt1_ids[pair[1]]]
                        query = f"record 1: {r1}, record 2: {r2}. Answer with True. or False."

                        resp = ollama.chat(
                            model=LLM,
                            messages=[{'role': 'user', 'content': query}],
                            options={'stop': ['\n', '.']},
                            stream=False
                        )

                        responses.append(resp['message']['content'])

                    end = time.time()

                    cp_df['responses'] = responses

                    time_seconds = end - start

                    good_behavior_rate = _get_good_behavior_response_rate(responses)

                    cp_df['responses'] = cp_df['responses'].apply(lambda x:
                                                                  'True' in str(x))

                    cp_df.to_csv(responses_filename, index=False)

                    precision, recall, f1 = _evaluate(gt_set, cp, cp_df['responses'].tolist())
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
                            'dataset_1': DIR,
                            'dataset': DIR,
                            'model': LLM,
                            'time (sec)': time_seconds,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'good_behavior_response_rate': good_behavior_rate,
                            "examples": examples,
                            'total_matches': filtered_cp_df.shape[0],
                            "conflicts": conflict_pct
                        }, index=[0]
                    )
                    ollama.delete(model=LLM)

                    header = not os.path.exists(f'post-review/results/{POST_REVIEW_EXP}/{BLOCKING_TYPE}/{DIR}.csv')
                    results_df.to_csv(f'post-review/results/{POST_REVIEW_EXP}/{BLOCKING_TYPE}/{DIR}.csv', mode='a+',
                                      index=False, header=header, float_format='%.2f')
        break



