import os
import pandas as pd
from gpt_utils import _evaluate, _load_dataset, _convert_to_numpy

if __name__ == "__main__":
    DIR = "D7"
    RESULTS = f"post-review/results/gpt/original/{DIR}.csv"
    RESPONSES = f"post-review/responses/original/{DIR}"
    BLOCKING_TYPE = "original"
    dt1_df, dt2_df, cp_df, gt_df, clean_files = _load_dataset(BLOCKING_TYPE,
                                                            DIR)
    dt_1, dt_2, cp, gt_set = _convert_to_numpy(dt1_df, dt2_df, cp_df, gt_df)

    df = pd.read_csv(RESULTS)

    print(df.head(10))

    precisions = []
    recalls = []
    f1s = []
    for _, row in df.iterrows():
        llm = row['model']
        examples = row['examples']

        responses_path = f"{RESPONSES}/{DIR}_{llm}_{examples}.csv"

        responses_df : pd.DataFrame = pd.read_csv(responses_path)
        if responses_df['responses'].dtype != bool:
            responses_df['responses'].apply(lambda x: True if 'True' in str(x) else False)
            responses_df['responses'] = responses_df['responses'].astype(bool)


        precision, recall, f1 = _evaluate(gt_set, cp, responses_df['responses'].tolist())

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    df['recall'] = recalls
    df['precision'] = precisions
    df['f1'] = f1s

    print(df.head(10))

    df.to_csv(RESULTS, index=False, float_format='%.2f')





