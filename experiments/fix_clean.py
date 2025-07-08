import pandas as pd

path = 'candidate_pairs'
files = ['D3', 'D5', 'D6', 'D7', 'D8']


placeholder = '__NaN__'



for file in files: 
    cp = pd.read_csv(f'{path}/{file}.csv')
    cp_columns = list(cp.columns)
    clean_files = [cl.replace("clean", "").replace(".csv", "") for cl in cp_columns]
    if file == 'D3':
        original_d1 = pd.read_csv(f'data_clean/{file}/{clean_files[0]}.csv', sep='#')             
        original_d2 = pd.read_csv(f'data_clean/{file}/{clean_files[1]}.csv', sep='#')             
        clean_d1 = pd.read_csv(f'data_clean/{file}/{clean_files[0]}clean.csv', sep='#')             
        clean_d2 = pd.read_csv(f'data_clean/{file}/{clean_files[1]}clean.csv', sep='#')             
    else:
        original_d1 = pd.read_csv(f'data_clean/{file}/{clean_files[0]}.csv', sep='|')             
        original_d2 = pd.read_csv(f'data_clean/{file}/{clean_files[1]}.csv', sep='|')             
        clean_d1 = pd.read_csv(f'data_clean/{file}/{clean_files[0]}clean.csv', sep='|')             
        clean_d2 = pd.read_csv(f'data_clean/{file}/{clean_files[1]}clean.csv', sep='|')
    mask = ~ (cp[cp_columns[0]].isin(clean_d1['id']) 
                & cp[cp_columns[1]].isin(clean_d2['id']))
    filtered_df = cp[mask]
    
    
    for index, row in filtered_df.iterrows():
        val1 = row[cp_columns[0]]
        val2 = row[cp_columns[1]]

        in_d1 = val1 in clean_d1['id'].values
        in_d2 = val2 in clean_d2['id'].values
        
        if not in_d1:
            d1_columns = list(original_d1.columns)
            
            
            cols_to_compare = [col for col in d1_columns[1:]]
            
            orig_row = original_d1[original_d1['id'] == val1]
            if not orig_row.empty:
                element = orig_row.iloc[0]
                # title_val = orig_row.iloc[0][d1_columns[1]]
                 # Find the first row in original_d1 where ALL other columns match
                match_row = original_d1[
                    (original_d1[cols_to_compare].fillna(placeholder) == element[cols_to_compare].fillna(placeholder)).all(axis=1)
                ]

                # match_row = original_d1[original_d1[d1_columns[1]] == title_val]
                if not match_row.empty:
                    val_first_1 = match_row.iloc[0]['id']

                    # Replace val1 in cp DataFrame
                    cp.at[index, cp_columns[0]] = val_first_1
            
        if not in_d2:
            d2_columns = list(original_d2.columns)
            cols_to_compare = [col for col in d2_columns[1:]]
            
            orig_row = original_d2[original_d2['id'] == val2]
            if not orig_row.empty:
                element = orig_row.iloc[0]
                # title_val = orig_row.iloc[0][d2_columns[1]]

                match_row = original_d2[
                    (original_d2[cols_to_compare].fillna(placeholder) == element[cols_to_compare].fillna(placeholder)).all(axis=1)
                ]
                
                if not match_row.empty:
                    val_first_1 = match_row.iloc[0]['id']

                    # Replace val1 in cp DataFrame
                    cp.at[index, cp_columns[1]] = val_first_1
    
    cnt=0              
    for index, row in cp.iterrows():
        val1 = row[cp_columns[0]]
        val2 = row[cp_columns[1]]

        in_d1 = val1 in clean_d1['id'].values
        in_d2 = val2 in clean_d2['id'].values
        
        if not (in_d1 and in_d2):
            print(file, val1, val2)
            exit(10)
            cnt+=1
    if cnt == 0:        
        print(f'{file}: {cnt}')
        cp.to_csv(f'{path}/{file}.csv', index=False)  
                


            
            
        # print(filtered_df)            
    
        



