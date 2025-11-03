import pandas as pd
import os

def preprocess_raw_dataset(source_domain, target_domain, source_path, target_path, k1=5, k2=5):
    # 读入gzip文件
    try:
        df_source = pd.read_csv(source_path, compression="gzip")
        df_target = pd.read_csv(target_path, compression="gzip")
        df_source = df_source.rename(
            columns={"user_id": "userID", "parent_asin": "itemID"}
        )
        df_target = df_target.rename(
            columns={"user_id": "userID", "parent_asin": "itemID"}
        )
    except Exception:
        df_source = pd.read_json(source_path, lines=True, compression="gzip")
        df_target = pd.read_json(target_path, lines=True, compression="gzip")
        df_source = df_source[["reviewerID", "asin", "overall", "unixReviewTime"]].rename(
            columns={"reviewerID": "userID", "asin": "itemID", "overall": "rating", "unixReviewTime": "timestamp"}
        )
        df_target = df_target[["reviewerID", "asin", "overall", "unixReviewTime"]].rename(
            columns={"reviewerID": "userID", "asin": "itemID", "overall": "rating", "unixReviewTime": "timestamp"}
        )

    # 筛选出两个域的重叠用户
    users_source = set(df_source["userID"].unique())
    users_target = set(df_target["userID"].unique())
    overlapped_users = users_source & users_target
    df_source = df_source[df_source["userID"].isin(overlapped_users)].reset_index(drop=True)
    df_target = df_target[df_target["userID"].isin(overlapped_users)].reset_index(drop=True)

    # k-core
    df_source, df_target = k_core(df_source, df_target, k1, k2)

    # 用户id映射
    assert set(df_source["userID"].unique()) == set(df_target["userID"].unique())
    assert df_source["userID"].nunique() > 0
    unique_users = sorted(set(df_source["userID"].unique()))
    user2id = {u: i for i, u in enumerate(unique_users)}
    df_source["userID"] = df_source["userID"].map(user2id)
    df_target["userID"] = df_target["userID"].map(user2id)
    # 源域物品id映射
    unique_items_source = sorted(df_source["itemID"].unique())
    item2id_source = {it: i for i, it in enumerate(unique_items_source)}
    df_source["itemID"] = df_source["itemID"].map(item2id_source)
    # 目标域物品id映射
    unique_items_target = sorted(df_target["itemID"].unique())
    item2id_target = {it: i for i, it in enumerate(unique_items_target)}
    df_target["itemID"] = df_target["itemID"].map(item2id_target)
    # 划分train/valid/test
    df_source = split(df_source)
    df_target = split(df_target)

    #保存为inter并输出结果
    if "Amazon2023" in source_path:
        out_dir = os.path.join("../data/Amazon2023", f"{source_domain}_{target_domain}")
    else:
        out_dir = os.path.join("../data/Amazon2014", f"{source_domain}_{target_domain}")
    os.makedirs(out_dir, exist_ok=True)
    source_file = os.path.join(out_dir, "source.inter")
    target_file = os.path.join(out_dir, "target.inter")
    df_source.to_csv(source_file, index=False, sep='\t')
    df_target.to_csv(target_file, index=False, sep='\t')

    msg = (
        "{}_{} dataset created!\n"
        "The number of user = {}\n"
        "The number of source item = {}, the density = {:.6f}\n"
        "The number of target item = {}, the density = {:.6f}\n"
    ).format(
        source_domain, target_domain,
        df_source["userID"].nunique(),
        df_source["itemID"].nunique(), df_source.shape[0] / (df_source["userID"].nunique() * df_source["itemID"].nunique()),
        df_target["itemID"].nunique(), df_target.shape[0] / (df_target["userID"].nunique() * df_target["itemID"].nunique())
    )
    print(msg)

def k_core(df_source, df_target, k1, k2):
    while True:
        n_users_before = df_source["userID"].nunique(), df_target["userID"].nunique()
        n_items_before = df_source["itemID"].nunique(), df_target["itemID"].nunique()

        # 用户在两个域的交互都要 >=k1
        user_counts_source = df_source["userID"].value_counts()
        user_counts_target = df_target["userID"].value_counts()
        valid_users = set(user_counts_source[user_counts_source >= k1].index) & \
                      set(user_counts_target[user_counts_target >= k1].index)

        df_source = df_source[df_source["userID"].isin(valid_users)].reset_index(drop=True)
        df_target = df_target[df_target["userID"].isin(valid_users)].reset_index(drop=True)

        # 2. 物品在各自域内的交互 >=k2
        item_counts_source = df_source["itemID"].value_counts()
        valid_items_source = set(item_counts_source[item_counts_source >= k2].index)
        df_source = df_source[df_source["itemID"].isin(valid_items_source)].reset_index(drop=True)

        item_counts_target = df_target["itemID"].value_counts()
        valid_items_target = set(item_counts_target[item_counts_target >= k2].index)
        df_target = df_target[df_target["itemID"].isin(valid_items_target)].reset_index(drop=True)

        # 如果用户和物品数量没有再变化，跳出循环
        n_users_after = df_source["userID"].nunique(), df_target["userID"].nunique()
        n_items_after = df_source["itemID"].nunique(), df_target["itemID"].nunique()

        if n_users_before == n_users_after and n_items_before == n_items_after:
            break

    return df_source, df_target


def split(df):
    df = df.sort_values(["userID", "timestamp"]).reset_index(drop=True)
    df["x_label"] = 0

    def assign_labels(group):
        n_items = len(group)
        if n_items < 10:
            group.iloc[-2, group.columns.get_loc("x_label")] = 1
            group.iloc[-1, group.columns.get_loc("x_label")] = 2
        else:
            val_test_len = int(n_items * 0.2)
            train_len = n_items - val_test_len
            val_len = val_test_len // 2
            test_len = val_test_len - val_len

            group.iloc[:train_len, group.columns.get_loc("x_label")] = 0
            group.iloc[train_len:train_len+val_len, group.columns.get_loc("x_label")] = 1
            group.iloc[train_len+val_len:train_len+val_len+test_len, group.columns.get_loc("x_label")] = 2
        return group

    df = df.groupby("userID", group_keys=False).apply(assign_labels)
    return df


if __name__ == '__main__':
    #Amazon2023 (5 core)
    sport_path = "../data/Amazon2023/Sports_and_Outdoors.csv.gz"
    clothing_path = "../data/Amazon2023/Clothing_Shoes_and_Jewelry.csv.gz"
    preprocess_raw_dataset("sport", "clothing", sport_path, clothing_path, 8, 7)

    toy_path = "../data/Amazon2023/Toys_and_Games.csv.gz"
    video_path = "../data/Amazon2023/Video_Games.csv.gz"
    preprocess_raw_dataset("toy", "video", toy_path, video_path, 4, 3)

    movie_path = "../data/Amazon2023/Movies_and_TV.csv.gz"
    cd_path = "../data/Amazon2023/CDs_and_Vinyl.csv.gz"
    preprocess_raw_dataset("movie", "cd", movie_path, cd_path, 6, 5)
    # preprocess_raw_dataset("movie", "video", movie_path, video_path, 4, 4)

    # # Amazon2014 (5 core)
    # sport_path = "../data/Amazon2014/reviews_Sports_and_Outdoors_5.json.gz"
    # clothing_path = "../data/Amazon2014/reviews_Clothing_Shoes_and_Jewelry_5.json.gz"
    # # preprocess_raw_dataset("Sport", "Clothing", sport_path, clothing_path, 0, 0)
    #
    # toy_path = "../data/Amazon2014/reviews_Toys_and_Games_5.json.gz"
    # video_path = "../data/Amazon2014/reviews_Video_Games_5.json.gz"
    # # preprocess_raw_dataset("Toy", "Video", toy_path, video_path, 0, 0)
    #
    # movie_path = "../data/Amazon2014/reviews_Movies_and_TV_5.json.gz"
    # cd_path = "../data/Amazon2014/reviews_CDs_and_Vinyl_5.json.gz"
    # preprocess_raw_dataset("Movie", "CD", movie_path, cd_path, 0, 0)
