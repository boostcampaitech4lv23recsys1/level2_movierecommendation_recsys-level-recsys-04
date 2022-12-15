import pandas as pd
from scipy.sparse import csr_matrix


def load_csv(data_name):
    """ 데이터 불러오는 함수 """
    df = pd.read_csv(data_name)
    return df


def get_factorized_index(args, input_df):
    # user, item mapping 작업
    user, user_map =  pd.factorize(input_df['user'], sort=True)
    input_df['user'] = user
    item, item_map = pd.factorize(input_df['item'], sort=True)
    input_df['item'] = item

    args.user_map = user_map
    args.item_map = item_map
    args.user_cnt = max(user) + 1
    args.item_cnt = max(item) + 1

    return input_df


# def get_user_seq(input_df):
#     user_seq = input_df.groupby('user')['item'].apply(list)
#     return user_seq
