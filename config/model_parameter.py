import tensorflow as tf


class model_parameter:

    def __init__(self):
        # Network parameters
        self.flags = tf.flags
        #self.flags.DEFINE_string('version', 'istsbp_end_2_end','model version')
        self.flags.DEFINE_string('version', 'bpr', 'model version')
        self.flags.DEFINE_string('checkpoint_path_dir', 'data/check_point/bisIE_adam_blocks2_adam_dropout0.5_lr0.0001/','directory of save model')
        self.flags.DEFINE_integer('hidden_units', 128, 'Number of hidden units in each layer')
        self.flags.DEFINE_integer('num_blocks', 1, 'Number of blocks in each attention')
        self.flags.DEFINE_integer('num_heads', 8, 'Number of heads in each attention')
        self.flags.DEFINE_integer('num_units', 128, 'Number of units in each attention')

        self.flags.DEFINE_float('dropout', 0.5, 'Dropout probability(0.0: no dropout)')
        self.flags.DEFINE_float('regulation_rate', 0.00005, 'L2 regulation rate')
        self.flags.DEFINE_integer('itemid_embedding_size', 64, 'Item id embedding size')
        self.flags.DEFINE_integer('cateid_embedding_size', 64, 'Cate id embedding size')

        self.flags.DEFINE_boolean('concat_time_emb', True, 'Concat time-embedding instead of Add')

        # 随机梯度下降sgd
        self.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop,sgd*)')
        self.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
        # 最大梯度渐变到5
        self.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
        # 训练批次32
        self.flags.DEFINE_integer('train_batch_size', 100, 'Training Batch size')
        # 测试批次128
        self.flags.DEFINE_integer('test_batch_size', 100, 'Testing Batch size')
        # 最大迭代次数
        self.flags.DEFINE_integer('max_epochs', 50, 'Maximum # of training epochs')
        # 每100个批次的训练状态
        self.flags.DEFINE_integer('display_freq', 10, 'Display training status every this iteration')
        self.flags.DEFINE_integer('eval_freq', 20, 'Display training status every this iteration')
        self.flags.DEFINE_integer('max_len', 150, 'max len of attention')
        self.flags.DEFINE_integer('global_step', 100, 'global_step to summery AUC')

        # Runtime parameters
        # self.flags.DEFINE_string('cuda_visible_devices', '0,1,2,3,4,5,6,7', 'Choice which GPU to use')
        self.flags.DEFINE_string('cuda_visible_devices', '2', 'Choice which GPU to use')
        self.flags.DEFINE_float('per_process_gpu_memory_fraction', 0.9,
                                'Gpu memory use fraction, 0.0 for allow_growth=True')
        # date process parameters
        self.flags.DEFINE_integer('gap_num', 6, 'sequence gap')
        self.flags.DEFINE_boolean('is_training', True, 'train of inference')
        self.flags.DEFINE_string('type', "Amazon", 'raw date type')
        self.flags.DEFINE_string('experiment_type', "pistrec", 'experiment date type, e.g. istsbp, pistrec')

        #istsbp experiment type:
        #long_term.short_term.vanilla,concatenate_long_short_vanilla,
        #concatenate_long_vanilla,sum_long_vanilla,sum_short_vanilla，sum_long_short_vanilla
        #sum_long_short
        self.flags.DEFINE_string('ISTSBP_type','all','long_term, short_term, long_short_term_vanilla, all')
        self.flags.DEFINE_string('ISTSBP_period_prediction','True','True for peroid prediction; False for Next-item prediction')
        self.flags.DEFINE_integer('ISTSBP_period',7,'If ISTSBP is for Peroid prediction, ISTSBP_period days is the prediction window')
        self.flags.DEFINE_string('PISTRec_type','hard','hard,soft')
        self.flags.DEFINE_string('lstur_type', 'longandshort', 'longandshort,long,short')
        # self.flags.DEFINE_string('experiment_type',"bpr" ,'experiment date type')

        #parameters about origin_data
        self.flags.DEFINE_boolean('init_origin_data', False, 'whewher to initialize the origin data')
        self.flags.DEFINE_integer('user_count_limit', 10000, "the limit of user")
        self.flags.DEFINE_string('causality', "random", "the mask method")
        self.flags.DEFINE_string('pos_embedding', "time", "the method to embedding pos")
        self.flags.DEFINE_integer('test_frac', 5, "train test radio")
        self.flags.DEFINE_float('mask_rate', 0.2, 'mask rate')
        self.flags.DEFINE_float('neg_sample_ratio', 5, 'negetive sample ratio')
        self.flags.DEFINE_string('raw_data_path', None, "raw data path")
        self.flags.DEFINE_string('raw_data_path_meta', None, 'raw meta path')


        self.flags.DEFINE_string('fine_tune_load_path', None, 'the check point paht for the fine tune mode ')
        #parameters about model
        # load type：full or fine_tune or from_scratch
        self.flags.DEFINE_string('load_type', "from_scratch", "the type of loading data")
        self.flags.DEFINE_boolean('draw_pic', False, "whether to drwa picture")
        self.flags.DEFINE_integer('top_k', 20, "evaluate recall ndcg for k users")
        self.flags.DEFINE_string('embedding_config_file', "data/config/embedding__dic.csv", "the embedding config")
        # self.flags.DEFINE_string('experiment_name', "TomSun_server1", "the expeiment")
        self.flags.DEFINE_string('experiment_name', "tianchi", "the expeiment")


    def get_parameter(self,type):

        if type == "tianchi":
            self.flags.FLAGS.version = "gen_pic"
            self.flags.FLAGS.type = "Tianchi"
            self.flags.FLAGS.test_frac = 10
            self.flags.FLAGS.num_blocks = 2
            self.flags.FLAGS.user_count_limit = 1000
            self.flags.FLAGS.checkpoint_path_dir = None
            self.flags.FLAGS.max_epochs = 6
            self.flags.FLAGS.load_type = "from_scratch"
            self.flags.FLAGS.raw_data_path = "data/raw_data/tianchi_raw_data.csv"
            self.flags.FLAGS.raw_data_path_meta = "data/raw_data/tianchi_raw_data.csv"
            self.flags.FLAGS.train_batch_size = 50
            self.flags.FLAGS.init_origin_data = False
            self.flags.FLAGS.eval_freq = 10
            self.flags.FLAGS.fine_tune_load_path = "data/check_point/Tianchi_bsbe_bisIE_adam_blocks2_adam_dropout0.5_lr0.0001"
            self.flags.FLAGS.experiment_type = "slirec"
            self.flags.FLAGS.lstur_type = 'pref'
            self.flags.FLAGS.pos_embedding = "time"
            self.flags.FLAGS.causality = "random"
            self.flags.FLAGS.embedding_config_file = "config/embedding__dic_action.csv"

        elif type == "amazon":
            self.flags.FLAGS.version = "gen_pic"
            self.flags.FLAGS.type = "Amazon"
            self.flags.FLAGS.test_frac = 10
            self.flags.FLAGS.num_blocks = 2
            self.flags.FLAGS.user_count_limit = 1000
            self.flags.FLAGS.checkpoint_path_dir = None
            self.flags.FLAGS.max_epochs = 6
            self.flags.FLAGS.load_type = "from_scratch"
            self.flags.FLAGS.raw_data_path = "data/raw_data/reviews_Electronics_5.json"
            self.flags.FLAGS.raw_data_path_meta = "data/raw_data/meta_Electronics.json"
            self.flags.FLAGS.train_batch_size = 50
            self.flags.FLAGS.init_origin_data = False
            self.flags.FLAGS.eval_freq = 10
            self.flags.FLAGS.fine_tune_load_path = "data/check_point/Tianchi_bsbe_bisIE_adam_blocks2_adam_dropout0.5_lr0.0001"
            self.flags.FLAGS.experiment_type = "dmpn"
            self.flags.FLAGS.lstur_type = 'pref'
            self.flags.FLAGS.pos_embedding = "time"
            self.flags.FLAGS.causality = "random"
            self.flags.FLAGS.embedding_config_file = "config/embedding__dic.csv"

        return self.flags



