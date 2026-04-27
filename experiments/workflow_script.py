from pyjedai.datamodel import Data
import pandas as pd
from pyjedai.joins import TopKJoin
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding

if __name__ == "__main__":


    data_list = []
    # datasets_list  = ['D2','D3','D5','D6','D7','D8'] 
    datasets_list  = ['10K', '50K']

    for dataset in datasets_list:

        candidate_pairs = f'data/candidate_pairs/original/{dataset}.csv'
        cp_df_with_rows = pd.read_csv(candidate_pairs)
        cp_columns = list(cp_df_with_rows.columns)
        clean_files = [cl.replace("clean", "").replace(".csv", "") for cl in cp_columns]
                    


        d1 = pd.read_csv(f"data/{dataset}/{dataset}full.csv", sep='|')
        
        d1.drop(columns=['Embedded Ag.Value', 'Clean Ag.Value', 'Embedded Clean Ag.Value'], inplace=True)
        d1.to_csv(f"data/{dataset}/{dataset}.csv", sep='|', index=False)
        # d2 = pd.read_csv(f"data/{dataset}/{clean_files[1]}clean.csv", sep='#')
        gt = pd.read_csv(f"data/{dataset}/{dataset}duplicates.csv", sep='|')
        


        gt.columns = ['D1', 'D2']
        # cp_df = pd.DataFrame({
        #             'D1': cp_df_with_rows[cp_columns[0]].map(d1['id']),
        #             'D2': cp_df_with_rows[cp_columns[1]].map(d2['id'])
        # })


        cp_df = cp_df_with_rows
        cp_df.columns = ['D1','D2']
        common = pd.merge(gt, cp_df)


        # cp = cp_df.to_numpy()
        cp = list((int(pair['D1']), int(pair['D2'])) for _,pair in cp_df.iterrows())

        gt_set = list((int(pair['D1']), int(pair['D2'])) for _,pair in gt.iterrows())
        
        data = Data(
            dataset_1=d1,
            id_column_name_1='Id',
            # id_column_name_2='id',
            # dataset_2=d2,
            ground_truth=gt
        )

        data_list.append((data, cp, gt_set))

    g_list = []

    join = TopKJoin(K=5, metric='cosine', tokenization='qgrams')

    for i in data_list:
        data = i[0]
        g = join.fit(data)
        g_list.append(g)  


    emb = EmbeddingsNNBlockBuilding(vectorizer='sdistilroberta', similarity_search='faiss')

    # g_list_emb = []
    for i in data_list:
        data = i[0]
        bl, g = emb.build_blocks(data=data, num_of_clusters=1, top_k=1,similarity_distance = 'cosine',with_entity_matching=True)
        g_list.append(g)     
    # for l in range(len(datasets_list)):

    data = data_list[0][0]
    cp = data_list[0][1]
    gt_set = data_list[0][2]
        
    dataset = datasets_list[0]

    for g in g_list:
        
    
        # print(data)
            
        #  Get edges sorted by weight in ascending order
        edges_by_weight = sorted(g.edges(data=True), key=lambda x: x[2]['weight'])

        # # Extract just the (u,v) pairs
        uv_pairs = [(u, v) for u, v, _ in edges_by_weight]
        # i = 0
        
        for index1, index2 in uv_pairs:
            if index1 < len(data.dataset_1):
                id1 = int(data._gt_to_ids_reversed_1[index1])
                id2 = int(data._gt_to_ids_reversed_1[index2])
            else:
                id2 = int(data._gt_to_ids_reversed_1[index1])
                id1 = int(data._gt_to_ids_reversed_1[index2])

            in_cp = (id1, id2) in cp or (id2, id1) in cp
            in_gt = (id1, id2) in gt_set or (id2, id1) in gt_set
            if in_cp and in_gt:
                my_str = f"[({id1}, {id2}),"
                break

        for index1, index2 in reversed(uv_pairs):
            if index1 < len(data.dataset_1):
                id1 = int(data._gt_to_ids_reversed_1[index1])
                id2 = int(data._gt_to_ids_reversed_1[index2])
            else:
                id2 = int(data._gt_to_ids_reversed_1[index1])
                id1 = int(data._gt_to_ids_reversed_1[index2])

            in_cp = (id1, id2) in cp or (id2, id1) in cp
            in_gt = (id1, id2) in gt_set or (id2, id1) in gt_set
            if in_cp and not in_gt:
                print(f"\"{dataset}\":{my_str} ({id1}, {id2})],")
                break




        for index1, index2 in reversed(uv_pairs):
            if index1 < len(data.dataset_1):
                id1 = int(data._gt_to_ids_reversed_1[index1])
                id2 = int(data._gt_to_ids_reversed_1[index2])
            else:
                id2 = int(data._gt_to_ids_reversed_1[index1])
                id1 = int(data._gt_to_ids_reversed_1[index2])

            in_cp = (id1, id2) in cp or (id2, id1) in cp
            in_gt = (id1, id2) in gt_set or (id2, id1) in gt_set
            if in_cp and in_gt:
                my_str = f"[({id1}, {id2}),"
                break

        for index1, index2 in uv_pairs:
            if index1 < len(data.dataset_1):
                id1 = int(data._gt_to_ids_reversed_1[index1])
                id2 = int(data._gt_to_ids_reversed_1[index2])
            else:
                id2 = int(data._gt_to_ids_reversed_1[index1])
                id1 = int(data._gt_to_ids_reversed_1[index2])

            in_cp = (id1, id2) in cp or (id2, id1) in cp
            in_gt = (id1, id2) in gt_set or (id2, id1) in gt_set
            if in_cp and not in_gt:
                print(f"\"{dataset}\":{my_str} ({id1}, {id2})],")
                break

    