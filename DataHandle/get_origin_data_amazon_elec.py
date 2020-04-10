import pandas as pd
import numpy as np
import os
from config.model_parameter import model_parameter
from DataHandle.get_originx_data_base import Get_origin_data_base
np.random.seed(1234)

class Get_amzon_data(Get_origin_data_base):
    def __init__(self, FLAGS):
        super(Get_amzon_data, self).__init__(FLAGS=FLAGS)
        self.data_path = "data/orgin_data/amazon.csv"
        self.raw_data_path = "data/raw_data/reviews_Electronics_5.json"
        self.raw_data_path_meta = "data/raw_data/meta_Electronics.json"

        if FLAGS.init_origin_data == True:
            if os.path.exists(self.data_path):
                self.origin_data = pd.read_csv(self.data_path)
            else:
                self.get_amazon_data()
        else:
            self.origin_data = pd.read_csv(self.data_path)

    def get_amazon_data(self):
        with open(self.raw_data_path, 'r') as fin:
            #确保字段相同
            resultList = []
            for line in fin:
                try:
                    tempDic = eval(line)
                    resultDic = {}
                    resultDic["user_id"] = tempDic["reviewerID"]
                    resultDic["item_id"] = tempDic["asin"]
                    resultDic["time_stamp"] = tempDic["unixReviewTime"]

                    resultList.append(resultDic)
                except Exception as e:
                    self.logger.info("Error！！！！！！！！！！！！")
                    self.logger.info(e)

            reviews_Electronics_df = pd.DataFrame(resultList)

        with open(self.raw_data_path_meta, 'r') as fin:
            resultList = []
            for line in fin:
                tempDic = eval(line)
                resultDic = {}
                resultDic["cat_id"] = tempDic["categories"][-1][-1]
                resultDic["item_id"] = tempDic["asin"]

                resultList.append(resultDic)

            meta_df = pd.DataFrame(resultList)

        reviews_Electronics_df = pd.merge(reviews_Electronics_df, meta_df, on="item_id")
        print(reviews_Electronics_df.shape)

        resultItemList = []

        def GetMostItemFrency(x):
            if x.shape[0] >= 30:
                resultDic = {}
                resultDic["item_id"] = x["item_id"].tolist()[0]
                resultItemList.append(resultDic)

        reviews_Electronics_df.groupby(["item_id"]).apply(lambda x: GetMostItemFrency(x))
        resultItemList = pd.DataFrame(resultItemList)
        # print(resultItemList)

        resultUserList = []

        def GetUserItemFrency(x):
            if x.shape[0] >= 15:
                resultDic = {}
                resultDic["user_id"] = x["user_id"].tolist()[0]
                resultUserList.append(resultDic)

        reviews_Electronics_df.groupby(["user_id"]).apply(lambda x: GetUserItemFrency(x))
        resultUserList = pd.DataFrame(resultUserList)
        # print(resultUserList)

        reviews_Electronics_df = pd.merge(reviews_Electronics_df, resultItemList, on="item_id")
        reviews_Electronics_df = pd.merge(reviews_Electronics_df, resultUserList, on="user_id")

        print(reviews_Electronics_df.shape)
        reviews_Electronics_df.to_csv(self.data_path, index=False, encoding="UTF8")
        self.origin_data = reviews_Electronics_df

if __name__ == '__main__':
    model_parameter_ins = model_parameter()
    experiment = model_parameter_ins.flags.FLAGS.experiment_name
    FLAGS = model_parameter_ins.get_parameter(experiment).FLAGS
    ins = Get_amzon_data(FLAGS=FLAGS)