from DataHandle.get_origin_data_taobao import Get_taobao_data
from DataHandle.get_origin_data_tmall import Get_tmall_data
from DataHandle.get_origin_data_amazon_elec import Get_amzon_data
from DataHandle.get_origin_data_movielen import Get_movie_data
from config.model_parameter import model_parameter
from util.model_log import create_log

class Get_origin_data():
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        log_ins = create_log()
        self.logger = log_ins.logger

        if self.FLAGS.type == 'taobao':
            self.get_origin_data_ins = Get_taobao_data(self.FLAGS)
        elif self.FLAGS.type == 'tmall':
            self.get_origin_data_ins = Get_tmall_data(self.FLAGS)
        elif self.FLAGS.type == 'amazon' \
                or self.FLAGS.type == 'beauty'\
                or self.FLAGS.type == 'kindle':
            self.get_origin_data_ins = Get_amzon_data(self.FLAGS)
        elif self.FLAGS.type == 'movie':
            self.get_origin_data_ins = Get_movie_data(self.FLAGS)


        self.origin_data = self.get_origin_data_ins.origin_data

    def getDataStatistics(self):

        df = self.origin_data
        user = set(df['user_id'].tolist())
        print("the user count is" + str(len(user)))
        item = set(df['item_id'].to_list())
        print("the item count is" + str(len(item)))
        category = set(df['cat_id'].to_list())
        print("the category count is" + str(len(category)))

        behavior_count = df.shape[0]
        print("the behavior count is" + str(behavior_count))

        behavior_per_user = df.groupby(by=["user_id"], as_index=False)["item_id"].count()
        behavior_per_user = behavior_per_user["item_id"].mean()
        print("The avg behavior of each user count is " + str(behavior_per_user))

        behavior_per_item = df.groupby(by=["item_id"], as_index=False)["user_id"].count()
        behavior_per_item = behavior_per_item["user_id"].mean()
        print("The avg behavior of each item count is " + str(behavior_per_item))

if __name__ == '__main__':
    model_parameter_ins = model_parameter()
    experiment = model_parameter_ins.flags.FLAGS.experiment_name
    FLAGS = model_parameter_ins.get_parameter(experiment).FLAGS
    ins = Get_origin_data(FLAGS=FLAGS)
    ins.getDataStatistics()