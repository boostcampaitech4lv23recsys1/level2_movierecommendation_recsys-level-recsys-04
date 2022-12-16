import random
import pandas as pd


def get_user_seq(args, data):
    user_seq = data.groupby('user')['item'].apply(list)
    users = []; items = []; answers = []

    for user, item in zip(user_seq.index, user_seq):
        item_len = len(item)
        item_set = set(item)

        neg_len = int(item_len * args.neg_ratio)
        for _ in range(neg_len):
            cur_item = random.randint(0, args.item_cnt-1)
            while cur_item in item_set:
                cur_item = random.randint(0, args.item_cnt-1)
            item.append(cur_item)
            item_set.add(cur_item)
        
        answer = [1] * item_len + [0] * neg_len
        user_list = [user] * (item_len + neg_len)

        users.extend(user_list)
        items.extend(item)
        answers.extend(answer)
    
    df = pd.DataFrame({'user': users, 'item': items, 'answer': answers})
    return df
