import pandas as pd
import numpy as np
import os

union_sym = " U "
intersection_sym = " ∩ "

if __name__ == "__main__":

    BLOCKING_TYPE = "standard_blocking"
    DIR = "D7"
    PATH = f"results/{BLOCKING_TYPE}/{DIR}.csv"

    df = pd.read_csv(PATH)
    df.drop(columns=['D1_conflicts','D2_conflicts'], inplace=True)

    conflicts = []
    
    

    for _, row in df.iterrows():
        llm = row['model']
        
        examples = row['examples']

        if union_sym in llm:
            llm = llm.replace(union_sym, "_union_")
        elif intersection_sym in llm:
            llm = llm.replace(intersection_sym, "_intersection_")
        responses_path = f'responses/{BLOCKING_TYPE}/{DIR}/{DIR}_{llm}_{examples}.csv'

        if '-z' in llm and not os.path.exists(responses_path):
            responses_path = f'responses/{BLOCKING_TYPE}/{DIR}/{DIR}_{llm}.csv'

        responses_df : pd.DataFrame = pd.read_csv(responses_path)

        if responses_df['responses'].dtype != bool:
            responses_df['responses'].apply(lambda x: True if 'True' in str(x) else False)
            responses_df['responses'] = responses_df['responses'].astype(bool)

        responses_df = responses_df[responses_df['responses'] == True]

        candidate_pairs = responses_df[["D1", "D2"]].to_numpy()

        u0, counts0 = np.unique(candidate_pairs[:, 0], return_counts=True)
        u1, counts1 = np.unique(candidate_pairs[:, 1], return_counts=True)

        u0 = u0[counts0 == 1]
        u1 = u1[counts1 == 1]

        total_pairs = candidate_pairs.shape[0]

        non_conflicts = np.isin(candidate_pairs[:, 0], u0) & np.isin(candidate_pairs[:, 1], u1)

        conflict_pct = (1 - non_conflicts.sum() / total_pairs) * 100 if total_pairs > 0 else 0.0
        conflicts.append(conflict_pct)



    for col in ['recall', 'precision', 'f1']:
        df[col] = df[col] * 100

    df['conflicts'] = conflicts
    print(df.head(10))
    FINAL_RESULTS = f"post-review/results/conflicts/{BLOCKING_TYPE}/{DIR}.csv"
    df.to_csv(FINAL_RESULTS, index=False, float_format='%.2f')


    exit(10)