from util.model_log import create_log
import pandas as pd
from config.model_parameter import model_parameter

class Get_origin_data_base():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

    def filter(self, data):
        item_filter = data.groupby("item_id").count()
        item_filter = item_filter[item_filter['user_id'] >= 30]
        data = data[data['item_id'].isin(item_filter.index)]
        user_filter = data.groupby("user_id").count()
        user_filter = user_filter[user_filter['item_id'] >= 20]
        data = data[data['user_id'].isin(user_filter.index)]
        if self.FLAGS.type == "taobao" or self.FLAGS.type == "tmall":
            resultUserList = []

            def buy_count(x):
                buy_count = x.loc[x["action_type"] == 2].shape[0]
                if buy_count >= 5:
                    resultDit = {}
                    resultDit["user_id"] = x["user_id"].tolist()[0]
                    resultUserList.append(resultDit)

            data.groupby("user_id").apply(lambda x: buy_count(x))
            resultUserList = pd.DataFrame(resultUserList)
            data = pd.merge(data, resultUserList, on="user_id")
        return data