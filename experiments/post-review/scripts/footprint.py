import psutil
import os
from time import sleep
import subprocess
import pandas as pd
from tqdm.auto import  tqdm
from gpt_utils import _load_dataset, _create_model, _convert_to_numpy
from examples import examples_dict_list as original_examples
from examples_blocking import examples_dict_list as standard_blocking_examples
import ollama

union_sym = " U "
intersection_sym = " ∩ "

def main(blocking_type, dir_):
    POST_REVIEW_EXP = "memory"
    dt1_df, dt2_df, cp_df, gt_df, clean_files = _load_dataset(blocking_type, dir_)
    dt_1, dt_2, cp, gt = _convert_to_numpy(dt1_df, dt2_df, cp_df, gt_df)
    PATH = f"post-review/results/conflicts/{blocking_type}/{dir_}.csv"
        
    RESULTS = f'post-review/results/{POST_REVIEW_EXP}/{blocking_type}/{dir_}.csv'
    RESPONSES = f"responses/original/{blocking_type}/{dir_}"
    
    examples_dict_list = standard_blocking_examples if blocking_type == 'standard_blocking' else original_examples
    prompts = {
        "p2" : (
            "You are given two record descriptions and your task is to identify "
            "if the records refer to the same entity or not. "
            "You must answer with just one word: "
            "True. if the records are referring to the same entity, "
            "False. if the records are referring to a different entity."
        )
    }
    
    df = pd.read_csv(PATH)
    df = df[~df['model'].str.contains('p1', na=False)]
    
    ram = []
    vram = []
    
    for _, row in df.iterrows():
        llm = row['model']
        if 'p1' in llm:
            continue

        dt1_ids = {int(dt1_df.at[i, 'id']): i for i in range(len(dt1_df))}
        dt2_ids = {int(dt2_df.at[i, 'id']): i for i in range(len(dt2_df))}

        examples = row['examples']
        
        examples_dict = examples_dict_list[examples]

        if union_sym in llm:
            llm = llm.replace(union_sym, "_union_")
        elif intersection_sym in llm:
            llm = llm.replace(intersection_sym, "_intersection_")
        responses_path = f'responses/{blocking_type}/{dir_}/{dir_}_{llm}_{examples}.csv'

        if '-z' in llm and not os.path.exists(responses_path):
            responses_path = f'responses/{blocking_type}/{dir_}/{dir_}_{llm}.csv'
        ll = llm.split('-')[0]
        suffix = llm.split('-')[1]
        prompt =  _create_model(llm, ll, prompts["p2"], suffix,
            examples_dict, dir_, dt_1, dt_2, dt1_ids, dt2_ids)
        ollama.create(model=llm, from_=ll, system=prompt)

        responses_df : pd.DataFrame = pd.read_csv(responses_path)
        
        mem = psutil.virtual_memory()
        ram_gb = mem.used / (1024 ** 3)
        

        responses = []

        for pair in tqdm(cp, desc="Processing", position=1, leave=False):
            dt1_id = pair[0]
            dt2_id = pair[1]

            r1 = dt_1[dt1_ids[pair[0]]]
            r2 = dt_2[dt2_ids[pair[1]]]
            query = f"record 1: {r1}, record 2: {r2}. Answer with True. or False."

            resp = ollama.chat(
                model=llm,
                messages=[{'role': 'user', 'content': query}],
                options={'stop': ['\n', '.']},
                stream=False
            )

            responses.append(resp['message']['content'])
            break


        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], 
            encoding='utf-8'
        )
        # If you have multiple GPUs, this returns a list of strings separated by newlines
        vram_mb = int(result.strip().split('\n')[0])
        vram_used_by_model_gb = vram_mb / 1024
        
        ollama.delete(model=llm)

        ram.append(ram_gb)
        vram.append(vram_used_by_model_gb)
    df['RAM-USAGE'] = ram
    df['VRAM-USAGE'] = vram
    df.to_csv(RESULTS, index=False, float_format='%.2f')    

if  __name__ == '__main__':
    for blocking_type in ['original', 'standard_blocking']:
        for dir_ in ['D2', 'D5', 'D6', 'D7', 'D8']:
            print(f"Processing {blocking_type} - {dir_}")
            main(blocking_type, dir_)
            print(f"Done {blocking_type} - {dir_}")
            

