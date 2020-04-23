import random
import numpy as np
import pickle
import pandas as pd
from sklearn import preprocessing
import os
import copy
import random
from util.model_log import create_log
from Prepare.mask_data_process import mask_data_process
np.random.seed(1234)


class prepare_data_base():
    """
    Get_train_test(type,origin_data,experiment_type)

    generate training set and testing set
    -----------
    :parameter
        type: "tianchi", "Amazon"
        origin_data: Get_origin_data.origin_data
        experiment_type: "BSBE", "lSTSBP", "Atrank"....
        gapnum: number of gaps, default = 6
        data_set_limit: the limit of the data set
        test_frac： train test radio
    """

    def __init__(self, FLAGS, origin_data):

        self.FLAGS = FLAGS
        self.length = []
        self.type = FLAGS.type
        self.user_count_limit = FLAGS.user_count_limit
        self.test_frac = FLAGS.test_frac
        self.experiment_type = FLAGS.experiment_type
        self.neg_sample_ratio = FLAGS.neg_sample_ratio
        self.max_len = FLAGS.max_len
        self.model = FLAGS.experiment_type

        #give the data whether to use action
        if self.type == "tmall" or self.type == "taobao":
            self.use_action = True
        else:
            self.use_action = False

        self.data_type_error = 0
        self.data_too_short = 0

        # give reserve reserve field
        self.offset = 3

        # give the random  target value
        self.target_random_value = self.offset - 2

        # make origin data dir
        self.dataset_path = 'data/training_testing_data/'
        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)
        if self.model == "bpr":
            self.dataset_class_path = self.dataset_path + self.type + "_" + self.model + "_" + \
                                      self.FLAGS.pos_embedding + "_" + self.FLAGS.causality + '_train_test_class.pkl'
        else:
            self.dataset_class_path = self.dataset_path + self.type + "_" + \
                                      self.FLAGS.pos_embedding + "_" + self.FLAGS.causality + '_train_test_class.pkl'

        self.mask_rate = self.FLAGS.mask_rate
        log_ins = create_log()
        self.logger = log_ins.logger

        # init or load
        if FLAGS.init_origin_data == True:
            self.origin_data = origin_data
            self.get_gap_list(FLAGS.gap_num)
            self.map_process()
            self.filter_repetition()

        # load data
        else:
            with open(self.dataset_class_path, 'rb') as f:
                data_dic = pickle.load(f)
                self.train_set = data_dic["train_set"]
                self.test_set = data_dic["test_set"]
                # dont't need too large data set
                if len(self.test_set) > 3500:
                    self.test_set = random.sample(self.test_set, 3500)

                self.item_count = data_dic["item_count"]
                self.user_count = data_dic["user_count"]
                self.category_count = data_dic["category_count"]
                # self.gap = data_dic["gap"]
                # self.item_category_dic = data_dic["item_category"]
                self.logger.info("load data finish")
                self.logger.info('Size of training set is ' + str(len(self.train_set)))
                self.logger.info('Size of testing set is ' + str(len(self.test_set)))
                del data_dic

        self.init_origin_data = FLAGS.init_origin_data

    #give the index of item and category
    def map_process(self):
        """
        Map origin_data to one-hot-coding except time.

        """

        item_le = preprocessing.LabelEncoder()
        user_le = preprocessing.LabelEncoder()
        cat_le = preprocessing.LabelEncoder()

        # get item id list
        item_id = item_le.fit_transform(self.origin_data["item_id"].tolist())
        item_id = item_id + self.offset
        self.item_count = len(set(item_id))

        # get user id list
        user_id = user_le.fit_transform(self.origin_data["user_id"].tolist())
        user_id = user_id + self.offset
        self.user_count = len(set(user_id))

        # get category id list
        cat_id = cat_le.fit_transform(self.origin_data["cat_id"].tolist())
        cat_id = cat_id + self.offset
        self.category_count = len(set(cat_id))

        self.item_category_dic = {}
        for i in range(0, len(item_id)):
            self.item_category_dic[item_id[i]] = cat_id[i]

        self.logger.warning("item Count :" + str(len(item_le.classes_)))
        self.logger.info("user count is " + str(len(user_le.classes_)))
        self.logger.info("category count is " + str(len(cat_le.classes_)))

        # _key:key的列表，_map:key的列表加编号
        self.origin_data['item_id'] = item_id
        self.origin_data['user_id'] = user_id
        self.origin_data['cat_id'] = cat_id

        # 根据reviewerID、unixReviewTime编号进行排序（sort_values：排序函数）
        self.origin_data = self.origin_data.sort_values(['user_id', 'time_stamp'])

        # 重新建立索引
        self.origin_data = self.origin_data.reset_index(drop=True)
        return self.user_count, self.item_count

    #choose one for the action which are too close
    def filter_repetition(self):
        pass


    def get_train_test(self):
        """
        Generate training set and testing set with the mask_rate.
        The training set will be stored in training_set.pkl.
        The testing set will be stored in testing_set.pkl.
        dataset_path: 'data/training_testing_data/'
        :param
            data_size: number of samples
        :returns
            train_set: (user_id, item_list, (factor1_list, factor2,..., factorn), masked_item, label）
            test_set: (user_id, item_list, (factor1, factor2,..., factorn), (masked_item_positive,masked_item_negtive)）
            e.g. Amazon_bsbe
            train_set: (user_id, item_list, (time_interval_list, category_list), masked_item, label）
            test_set: (user_id, item_list,(time_interval_list, category_list), (masked_item_positive,masked_item_negtive)）
            e.g. Amazon_bsbe
            train_set: (user_id, item_list, (time_interval_list, category_list, action_list), masked_item, label）
            test_set: (user_id, item_list, (time_interval_list, category_list, action_list), (masked_item_positive,masked_item_negtive)）

        """
        if self.init_origin_data == False:
            return self.train_set, self.test_set

        self.data_set = []
        self.now_count = 0
        self.origin_data.groupby(["user_id"]).filter(lambda x: self.data_handle_process(x))
        self.format_train_test()

        random.shuffle(self.train_set)
        random.shuffle(self.test_set)

        self.logger.info('Size of training set is ' + str(len(self.train_set)))
        self.logger.info('Size of testing set is ' + str(len(self.test_set)))
        self.logger.info('Data type error size  is ' + str(self.data_type_error))
        self.logger.info('Data too short size is ' + str(self.data_too_short))


        with open(self.dataset_class_path, 'wb') as f:

            data_dic = {}
            data_dic["train_set"] = self.train_set
            data_dic["test_set"] = self.test_set
            data_dic["item_count"] = self.item_count
            data_dic["user_count"] = self.user_count
            data_dic["category_count"] = self.category_count
            # data_dic["gap"] = self.gap
            # data_dic["item_category"] = self.item_category_dic
            pickle.dump(data_dic, f, pickle.HIGHEST_PROTOCOL)

        return self.train_set, self.test_set

    def data_handle_process_base(self, x):

        behavior_seq = copy.deepcopy(x)
        behavior_seq = behavior_seq.sort_values(by=['time_stamp'], na_position='first')
        behavior_seq = behavior_seq.reset_index(drop=True)
        columns_value = behavior_seq.columns.values.tolist()
        if "user_id" not in columns_value:
            self.data_type_error = self.data_type_error + 1
            return

        pos_list = behavior_seq['item_id'].tolist()  # asin属性的值
        length = len(pos_list)

        #limit length
        if length < 20:
            self.data_too_short = self.data_too_short + 1
            return

        if length > self.max_len:
            behavior_seq = behavior_seq.tail(self.max_len)

        # user limit
        if self.now_count > self.user_count_limit:
            return

        self.now_count = self.now_count + 1

        # test
        behavior_seq = behavior_seq.reset_index(drop=True)

        return behavior_seq


    def data_handle_process(self, x):
        pass

    def format_train_test(self):
        # format like this
        # data_set = self.data_set
        # for i in range(len(data_set)):
        #     if self.experiment_type == "bsbe" or self.experiment_type == "istsbp":
        #         if i % self.test_frac == 0:
        #             test_set.append(data_set[i])
        #         else:
        #             train_set.append(data_set[i])
        #
        #     if self.experiment_type == "atrank" or self.experiment_type == "BisIE":
        #         if i % self.test_frac == 0:
        #             test_set.append((data_set[i][0], data_set[i][1], data_set[i][2], data_set[i][3][0], 1))
        #             test_set.append((data_set[i][0], data_set[i][1], data_set[i][2], data_set[i][3][1], 0))
        #         else:
        #             train_set.append((data_set[i][0], data_set[i][1], data_set[i][2], data_set[i][3][0], 1))
        #             train_set.append((data_set[i][0], data_set[i][1], data_set[i][2], data_set[i][3][1], 0))
        #
        #     if self.experiment_type == "bpr":
        #         if i % self.test_frac == 0:
        #             test_set.append((data_set[i][0], data_set[i][1][0], data_set[i][1][1]))
        #         else:
        #             train_set.append((data_set[i][0], data_set[i][1][0], data_set[i][1][1]))
        #
        #     if self.experiment_type == "rnn_att":
        #         if i % self.test_frac == 0:
        #             test_set.append((data_set[i][0], data_set[i][1], data_set[i][2][1], data_set[i][3][0], 1))
        #             test_set.append((data_set[i][0], data_set[i][1], data_set[i][2][1], data_set[i][3][1], 0))
        #         else:
        #             train_set.append((data_set[i][0], data_set[i][1], data_set[i][2][1], data_set[i][3][0], 1))
        #             train_set.append((data_set[i][0], data_set[i][1], data_set[i][2][1], data_set[i][3][1], 0))

        pass


    def get_gap_list(self, gapnum):
        gap = []
        for i in range(1, gapnum):
            if i == 1:
                gap.append(60)
            elif i == 2:
                gap.append(60 * 60)
            else:
                gap.append(3600 * 24 * np.power(2, i - 3))

        self.gap = np.array(gap)