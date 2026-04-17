from gpt_utils import (_load_dataset,
                       _get_responses_df,
                       _load_weights)


if __name__ == '__main__':
    BLOCKING_TYPE = 'standard_blocking'
    DIR = 'D8'
    POST_REVIEW_EXP = "gpt"
    dt1_df, dt2_df, cp_df, gt_df, clean_files = _load_dataset(BLOCKING_TYPE, DIR)


    RESULTS = f'post-review/results/{POST_REVIEW_EXP}/{BLOCKING_TYPE}/{DIR}.csv'

    s_weights = _load_weights(BLOCKING_TYPE, DIR, 's-weight')
    _weights = _load_weights(BLOCKING_TYPE, DIR, 'weight')
    for responses_df, responses_path, _ in _get_responses_df(RESULTS,
                                      BLOCKING_TYPE, DIR):

        responses_df['s-weight'] = s_weights
        responses_df['weight'] = _weights

        if 'Unnamed: 0' in set(responses_df.columns):
            responses_df.drop(columns=['Unnamed: 0'], inplace=True)

        responses_df.to_csv(responses_path, index=False, header=True, mode='w+')

