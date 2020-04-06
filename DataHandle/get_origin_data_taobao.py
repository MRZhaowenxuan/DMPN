import pandas as pd
import numpy as np
from config.model_parameter import model_parameter
np.random.seed(1234)
import time
import os

class Get_taobao_data():
    def __init__(self, FLAGS):
        self.data_path = "../data/orgin_data/taobao.csv"

        if FLAGS.init_origin_data == True:
            self.taobao_data = pd.read_csv("../data/raw_data/taobao/UserBehavior.csv", header=None,
                                           names=["user_id", "item_id", "item_category", "action_type", "time"])
            self.get_taobao_data()
        else:
            self.origin_data = pd.read_csv(self.data_path)

    def get_taobao_data(self):
        self.taobao_data = self.taobao_data[["user_id", "item_id", "item_category", "time"]]
        self.taobao_data = self.taobao_data.rename(columns={"item_category": "cat_id",
                                                            "time": "time_stamp"})
        self.filtered_taobao_data = self.filter(self.taobao_data)
        self.filtered_taobao_data.to_csv(self.data_path, encoding="UTF8", index=False)
        self.origin_data = self.filtered_taobao_data

    def filter(self, data):
        item_filter = data.groupby("item_id").count()
        item_filter = item_filter[item_filter['user_id'] >= 30]
        data = data[data['item_id'].isin(item_filter.index)]
        user_filter = data.groupby("user_id").count()
        data = data[data['user_id'].isin(user_filter.index)]
        return data

if __name__ == '__main__':
    model_parameter_ins = model_parameter()
    experiment = model_parameter_ins.flags.FLAGS.experiment_name
    FLAGS = model_parameter_ins.get_parameter(experiment).FLAGS
    ins = Get_taobao_data(FLAGS=FLAGS)