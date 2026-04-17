import pandas as pd
import ollama
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def _load_dataset(blocking_type, dir_):
    candidate_pairs = f'data/candidate_pairs/{blocking_type}/{dir_}.csv'
    cp_df_with_rows = pd.read_csv(candidate_pairs)
    cp_columns = list(cp_df_with_rows.columns)
    clean_files = [cl.replace("clean", "").replace(".csv", "") for cl in cp_columns]

    # data_dir = "giannis"
    dataset_1 = f'data/{dir_}/{clean_files[0]}clean.csv'
    dataset_2 = f'data/{dir_}/{clean_files[1]}clean.csv'
    groundtruth = f'data/{dir_}/gtclean.csv'


    sep = '|' if dir_!='D3' else '#'

    # read the files (Edit sep if needed)
    dt1_df = pd.read_csv(dataset_1, sep=sep)
    dt2_df = pd.read_csv(dataset_2, sep=sep)

    gt_df = pd.read_csv(groundtruth, sep=sep)

    dt1_df.rename(columns={dt1_df.columns[1]: 'title', dt1_df.columns[2]: 'name'}, inplace=True)
    dt2_df.rename(columns={dt2_df.columns[1]: 'title', dt2_df.columns[2]: 'name'}, inplace=True)


    dt1_df['title'] = dt1_df['title'].fillna('')
    dt1_df['name'] = dt1_df['name'].fillna('')

    dt2_df['title'] = dt2_df['title'].fillna('')
    dt2_df['name'] = dt2_df['name'].fillna('')


    print(dt1_df[['title','name']].head(10))

    if blocking_type == 'original':
        cp_df = pd.DataFrame({
            'D1': cp_df_with_rows[cp_columns[0]].map(dt1_df['id']),
            'D2': cp_df_with_rows[cp_columns[1]].map(dt2_df['id'])
        })
    else:
        cp_df = cp_df_with_rows
        cp_df.columns = ['D1','D2']

    return dt1_df, dt2_df, cp_df, gt_df, clean_files


def _create_model(llm, ll, prompt, suffix,
            examples_dict, dir_, dt1_df, dt2_df, dt1_ids, dt2_ids):
    if suffix != 'z':
        true_flag = True
        true_example = []
        false_example = []
        for pair in examples_dict[dir_]:
            r1  = f"Title: {dt1_df.at[dt1_ids[pair[0]], 'title']}, \
            Associated Name: {dt1_df.at[dt1_ids[pair[0]], 'name']}"

            r2 = f"Title: {dt2_df.at[dt2_ids[pair[1]], 'title']}, \
            Associated Name: {dt2_df.at[dt2_ids[pair[1]], 'name']}"

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

    ollama.create(model=llm, from_=ll, system=prompt)
    return prompt


def _convert_to_numpy(dt1_df, dt2_df, cp_df, gt_df):
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
    return dt1, dt2, cp, gt_set


def _get_response_time(start, end):
    # model's response time
    time_seconds = end - start
    hours, remainder = divmod(time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("Response Time: {:02}:{:02}:{:.2f}".
        format(int(hours), int(minutes), seconds))

def _get_good_behavior_response_rate(responses):
    good_responses = sum(response == 'True' or response == 'False' for response in responses)
    good_behavior_rate = good_responses / len(responses) if len(responses) > 0 else 0
    print("Good Behavior Response Rate:", good_behavior_rate)
    return good_behavior_rate

def _evaluate(gt_set, cp, responses):

    true_labels = [1 if tuple(pair) in gt_set else 0 for pair in cp]
    predicted_labels = [1 if resp else 0 for resp in responses]

    precision = precision_score(true_labels, predicted_labels) * 100
    recall = recall_score(true_labels, predicted_labels) * 100
    f1 = f1_score(true_labels, predicted_labels) * 100
    return precision, recall, f1



#                     # model's responses
#                     for i in range(len(responses)):
#                         if responses[i] != 'True' and responses[i] != 'False':
#                             responses[i] = 'False'

#                     if 'ft' in llm or 'tf' in llm:
#                         responses_filename = f"{dataset_dir}/{llm}_responses.txt"
#                         with open(responses_filename, 'w') as file:
#                             for response in responses:
#                                 file.write(response + '\n')
#                         print(f"Responses saved to {responses_filename} for union/intersection.")
#     if 'total_matches' not in list(results_df.columns):
#         total_matches = []
#         d1_conf = []
#         d2_conf = []

#         for _, row in results_df.iterrows():
#             llm = row['model']
#             examples = row['examples']
#             responses_path = f'responses/{candidate_pairs_dir}/{dataset}/{dataset}_{llm}_{examples}.csv'

#             if '-z' in llm and not os.path.exists(responses_path):
#                 responses_path = f'responses/{candidate_pairs_dir}/{dataset}/{dataset}_{llm}.csv'

#             responses_df = pd.read_csv(responses_path)

#             if isinstance(responses_df.iloc[0]['responses'], str):
#                 filtered_cp_df = responses_df[responses_df['responses'] == 'True']
#             else:
#                 filtered_cp_df = responses_df[responses_df['responses'] == True]

#             d1_list = filtered_cp_df['D1'].to_list()
#             d2_list = filtered_cp_df['D2'].to_list()
#             d1_set = set(d1_list)
#             d2_set = set(d2_list)

#             total_matches.append(len(d1_list))
#             d1_conf.append((len(d1_list) - len(d1_set))/len(d1_list))
#             d2_conf.append((len(d2_list) - len(d2_set))/len(d2_list))

#         results_df['total_matches'] = total_matches
#         results_df["D1_conflicts"] = d1_conf
#         results_df['D2_conflicts'] = d2_conf
#         results_df.to_csv(f'results/{candidate_pairs_dir}/{dataset}.csv', mode='w+', index=False, header=True)








#     ui_fun(dataset, candidate_pairs_dir)


# for dataset in ['D2', 'D5', 'D6', 'D7', 'D8' ]:
#     for candidate_pairs_dir in ["original", "standard_blocking"]:

#     # candidate_pairs_dir = "original"
#         ui_fun(dataset, candidate_pairs_dir)



# # for candidate_pairs_dir in ["original", "standard_blocking"]:
#     # dataset = 'D3'
#     # (dataset, candidate_pairs_dir)


