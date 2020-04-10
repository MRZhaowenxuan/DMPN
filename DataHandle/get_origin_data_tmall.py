import pandas as pd
import numpy as np
import os
import time
from config.model_parameter import model_parameter
from DataHandle.get_originx_data_base import Get_origin_data_base
np.random.seed(1234)

class Get_tmall_data(Get_origin_data_base):
    def __init__(self, FLAGS):
        super(Get_tmall_data, self).__init__(FLAGS=FLAGS)
        self.data_path = "data/orgin_data/tmall.csv"

        if FLAGS.init_origin_data == True:
            if os.path.exists(self.data_path):
                self.origin_data = pd.read_csv(self.data_path)
            else:
                self.tmall_data = pd.read_csv("data/raw_data/tianchi_raw_data.csv")
                self.get_tmall_data()
        else:
            self.origin_data = pd.read_csv(self.data_path)

    def get_tmall_data(self):
        def time2stamp(time_s):
            try:
                time_s = '2015' + str(time_s)
                time_s = time.strptime(time_s, "%Y%m%d")
                stamp = int(time.mktime(time_s))
                return stamp
            except Exception as e:
                print(time_s)
                print(e)
        self.tmall_data = self.tmall_data[["user_id", "item_id", "cat_id", "time_stamp", "action_type"]]
        self.tmall_data['time_stamp'] = self.tmall_data['time_stamp'].apply(lambda x: time2stamp(x))
        self.filtered_tmall_data = self.filter(self.tmall_data)
        self.filtered_tmall_data.to_csv(self.data_path, encoding="UTF8", index=False)
        self.origin_data = self.filtered_tmall_data


if __name__ == '__main__':
    model_parameter_ins = model_parameter()
    experiment = model_parameter_ins.flags.FLAGS.experiment_name
    FLAGS = model_parameter_ins.get_parameter(experiment).FLAGS
    ins = Get_tmall_data(FLAGS=FLAGS)