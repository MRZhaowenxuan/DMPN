import tensorflow as tf
from util.read_embedding_dic import embedding_csv_dic
import numpy as np
import copy
from Embedding.base_embedding import Base_embedding


class No_embedding(Base_embedding):

    def __init__(self,is_training = True,config_file = None):

        super(No_embedding, self).__init__(is_training,config_file)  # 此处修改了
        #self.init_placeholders()


    def init_placeholders(self):

        with tf.variable_scope("input_layer"):

            # [B] user id
            self.user_id = tf.placeholder(tf.int32, [None, ],name = "user")
            # [B] item id
            self.item_list = tf.placeholder(tf.int32, [None,None],name = "item_seq")

            #positive behavior item id
            self.item_positive = tf.placeholder(tf.int32, [None], name= 'item_positive')
            #negative behavior item id
            self.item_negative = tf.placeholder(tf.int32, [None,None], name= 'item_negative')

            self.seq_length = tf.placeholder(tf.int32, [None,],name = "seq_length")
            # index
            self.mask_index = tf.placeholder(tf.int32,[None,1], name='mask_index')
            #self.label_ids = tf.placeholder(tf.int32,[None,1], name='label_ids')


            # [B] item label
            #self.positive_y = tf.placeholder(tf.float32, [None, 1],name = "positive_label")
            #self.negative_y = tf.placeholder(tf.float32, [None,1], name="negative_label")

    def get_embedding(self,num_units):

        for key,values in self.embedding_dic.items():
            array_dim = values[0]
            embedding_dim = values[1]
            total_count = values[2]

            if key == "item":
                self.item_emb_lookup_table = self.init_embedding_lookup_table(name="item", total_count=total_count,
                                                                              embedding_dim = embedding_dim,
                                                                              is_training=self.is_training)
                tf.summary.histogram('item_emb_lookup_table', self.item_emb_lookup_table)
                behavior_seq_embedding_result = tf.nn.embedding_lookup(self.item_emb_lookup_table, self.item_list)
                behavior_positive_embedding_result = tf.nn.embedding_lookup(self.item_emb_lookup_table, self.item_positive)
                behavior_negative_embedding_result = tf.nn.embedding_lookup(self.item_emb_lookup_table, self.item_negative)


        behavior_seq_embedding_result_dense      = tf.layers.dense(behavior_seq_embedding_result, num_units,
                                                                   activation=tf.nn.relu,use_bias=False,
                                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
                                                                   name='dense4emb')# (N, T_q, C)
        behavior_positive_embedding_result_dense = tf.layers.dense(behavior_positive_embedding_result, num_units,
                                                                   activation=tf.nn.relu,use_bias=False,
                                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
                                                                   name='dense4emb',reuse=True)
        behavior_negative_embedding_result_dense = tf.layers.dense(behavior_negative_embedding_result, num_units,
                                                                   activation=tf.nn.relu,use_bias=False,
                                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
                                                                   name='dense4emb',
                                                                   reuse=True) # (N, T_q, C)



        return  behavior_seq_embedding_result, behavior_positive_embedding_result, \
                behavior_negative_embedding_result,\
                behavior_seq_embedding_result_dense, \
                behavior_positive_embedding_result_dense, \
                behavior_negative_embedding_result_dense,\
                self.mask_index,self.item_positive,self.seq_length



    def tranform_list_ndarray(self,deal_data,max_len,index):

        result = np.zeros([len(self.batch_data),max_len],np.float)

        k = 0
        for t in deal_data:
            for l in range(len(t[1])):
                result[k][l] = t[k][l]
            k += 1

        return result


    def concat_time_emb(self,item_seq_emb,):

        if self.config['concat_time_emb'] == True:
            t_emb = tf.one_hot(self.hist_t, 12, dtype=tf.float32)
            item_seq_emb = tf.concat([item_seq_emb, t_emb], -1)
            item_seq_emb = tf.layers.dense(item_seq_emb, self.config['hidden_units'])
        else:
            t_emb = tf.layers.dense(tf.expand_dims(self.hist_t, -1),
                                    self.config['hidden_units'],
                                    activation=tf.nn.tanh)
            item_seq_emb += t_emb

        return item_seq_emb

    #batch format   (user.item,factor_list,item_target.y...)
    def make_feed_dic(self,batch_data):
        self.batch_data = batch_data
        feed_dic = {}
        self.result_item = []
        seq_length = []
        index = 0
        #for target
        for key, values in self.embedding_dic.items():
            result_data_list = []
            result_data_pos_target = []
            result_data_neg_target = []
            for onedata in self.batch_data:
                if key == "item":
                    one_list = onedata[1]
                    self.result_item.append(onedata[4][0][0])
                    seq_length.append(len(one_list))

                max_len = int(values[3])

                #padding
                if len(one_list) < max_len:
                    padding_list = [0] * (max_len - len(one_list))
                    one_list = one_list + padding_list

                if len(one_list) > max_len:
                    one_list = one_list[:max_len]

                result_data_list.append(one_list)
                result_data_pos_target.append(onedata[4][0][index])

                #neg list
                neg_temp_list = []
                for neg_detail in onedata[4][1]:
                    neg_temp_list.append(neg_detail[index])

                result_data_neg_target.append(neg_temp_list)

            index = index + 1
            result_data_array = copy.deepcopy(np.array(result_data_list))
            result_data_pos_target_array = copy.deepcopy(np.array(result_data_pos_target))
            result_data_neg_target_array = copy.deepcopy(np.array(result_data_neg_target))

            key_list = key + "_list"
            if key == "item":
                now_placeholder = self.item_list

            feed_dic[now_placeholder] = result_data_array

            key_pos_target_id = key + "_positive"
            now_placeholder_target = getattr(self, key_pos_target_id)
            feed_dic[now_placeholder_target] = result_data_pos_target_array

            key_neg_target_id = key + "_negative"
            now_placeholder_target = getattr(self, key_neg_target_id)
            feed_dic[now_placeholder_target] = result_data_neg_target_array

            # # item_targe placeholder
        # feed_dic[self.item_target_id] = np.array([i[3] for i in batch_data])

        label_list = np.array([[i[3] for i in batch_data]]).T
        #y placeholder
        feed_dic[self.mask_index] = label_list
        feed_dic[self.seq_length] = np.array(seq_length)
        return feed_dic






