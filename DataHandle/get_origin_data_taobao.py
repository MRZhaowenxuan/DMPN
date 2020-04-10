import pandas as pd
import numpy as np
import os
from config.model_parameter import model_parameter
from DataHandle.get_originx_data_base import Get_origin_data_base
np.random.seed(1234)

class Get_taobao_data(Get_origin_data_base):
    def __init__(self, FLAGS):
        super(Get_taobao_data, self).__init__(FLAGS=FLAGS)
        self.data_path = "data/orgin_data/taobao.csv"

        if FLAGS.init_origin_data == True:
            if os.path.exists(self.data_path):
                self.origin_data = pd.read_csv(self.data_path)
            else:
                self.taobao_data = pd.read_csv("data/raw_data/taobao_raw_data.csv", header=None,
                                               names=["user_id", "item_id", "item_category", "action_type", "time"])
                self.taobao_data.action_type[self.taobao_data['action_type'] == 'pv'] = 0
                self.taobao_data.action_type[self.taobao_data['action_type'] == 'buy'] = 2
                self.taobao_data.action_type[self.taobao_data['action_type'] == 'cart'] = 1
                self.taobao_data.action_type[self.taobao_data['action_type'] == 'fav'] = 3
                self.get_taobao_data()
        else:
            self.origin_data = pd.read_csv(self.data_path)

    def get_taobao_data(self):
        self.taobao_data = self.taobao_data[["user_id", "item_id", "item_category", "time", "action_type"]]
        self.taobao_data = self.taobao_data.rename(columns={"item_category": "cat_id",
                                                            "time": "time_stamp"})
        self.filtered_taobao_data = self.filter(self.taobao_data)
        self.filtered_taobao_data.to_csv(self.data_path, encoding="UTF8", index=False)
        self.origin_data = self.filtered_taobao_data


if __name__ == '__main__':
    model_parameter_ins = model_parameter()
    experiment = model_parameter_ins.flags.FLAGS.experiment_name
    FLAGS = model_parameter_ins.get_parameter(experiment).FLAGS
    ins = Get_taobao_data(FLAGS=FLAGS)