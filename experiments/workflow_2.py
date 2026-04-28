import pandas as pd
from pyjedai.datamodel import Data
from pyjedai.vector_based_blocking import EmbeddingsNNBlockBuilding

if __name__ == "__main__":


    dataset = "10K"

    type_of_blocking = "standard_blocking"
    candidate_pairs = f'data/candidate_pairs/{type_of_blocking}/{dataset}.csv'
    cp_df_with_rows = pd.read_csv(candidate_pairs, header=None)
    cp_df_with_rows.columns = ['id1', 'id2']

    d1 = pd.read_csv(f"data/{dataset}/{dataset}.csv", sep='|')

    gt = pd.read_csv(f"data/{dataset}/{dataset}duplicates.csv", sep='|')



    gt.columns = ['D1', 'D2']


    cp_df = cp_df_with_rows


    cp_df.columns = ['D1','D2']

    valid_ids = pd.concat([cp_df['D1'], cp_df['D2']]).unique()
    gt_ids = pd.concat([gt['D1'], gt['D2']]).unique()
    valid_ids = set(valid_ids) & set(gt_ids)

    d1 = d1[d1['Id'].isin(valid_ids)]

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


    g_list = []

    # join = TopKJoin(K=5, metric='cosine', tokenization='qgrams')
    # g = join.fit(data)
    # edges_by_weight = sorted(g.edges(data=True), key=lambda x: x[2]['weight'])


    emb = EmbeddingsNNBlockBuilding(vectorizer='sdistilroberta', similarity_search='faiss')
    bl, g = emb.build_blocks(data=data, num_of_clusters=1, top_k=1,similarity_distance = 'cosine',with_entity_matching=True)
    edges_by_weight = sorted(g.edges(data=True), key=lambda x: x[2]['weight'])

    print("TRUE Positives")

    index_1, index_2 = edges_by_weight[-1][0], edges_by_weight[-1][1]

    print(data._gt_to_ids_reversed_1[index_1], data._gt_to_ids_reversed_1[index_2])

    index_1, index_2 = edges_by_weight[0][0], edges_by_weight[0][1]
    print(data._gt_to_ids_reversed_1[index_1], data._gt_to_ids_reversed_1[index_2])
