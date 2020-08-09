import tensorflow as tf
import numpy as np
import copy
from Embedding.base_embedding import Base_embedding


class Lstur_embedding(Base_embedding):

    def __init__(self, is_training=True, config_file=None,
                 user_count=100, item_count=100, category_count=100, max_len=50):

        super(Lstur_embedding, self).__init__(is_training, config_file)  # 此处修改了
        #self.init_placeholders()
        self.user_count = user_count
        self.item_count = item_count
        self.category_count = category_count
        self.action_count = 8
        self.max_len = max_len

    def init_placeholders(self):

        with tf.variable_scope("input_layer"):

            # [B] user id
            self.user_id = tf.placeholder(tf.int32, [None, ], name="user")
            # [B] item id
            self.item_list = tf.placeholder(tf.int32, [None, None], name="item_seq")

            #positive behavior item id
            self.item_positive = tf.placeholder(tf.int32, [None], name='item_positive')
            #negative behavior item id
            self.item_negative = tf.placeholder(tf.int32, [None, None], name='item_negative')

            self.seq_length = tf.placeholder(tf.int32, [None, ], name="seq_length")
            # index
            self.mask_index = tf.placeholder(tf.int32, [None, 1], name='mask_index')

            self.time_interval_list = tf.placeholder(tf.float32, [None, None], name='time_interval_list')
            self.time_list = tf.placeholder(tf.float32, [None, None], name='time_list')
            self.pos_last_list = tf.placeholder(tf.float32, [None, None], name='pos_last_list')

            for key in self.embedding_dic.keys():
                key_id = key + "_list"
                if key != "item":
                    temp_placeholder = tf.placeholder(tf.int32, [None, None], name=key_id)
                    setattr(self, key_id, temp_placeholder)
                    #target embedding placeholder
                    key_target_id = key + "_positive"
                    temp_placeholder = tf.placeholder(tf.int32, [None, ], name=key_target_id)
                    setattr(self, key_target_id, temp_placeholder)
                    # key_target_id = key + "_negative"
                    # temp_placeholder = tf.placeholder(tf.int32, [None,None], name=key_target_id)
                    # setattr(self, key_target_id, temp_placeholder)

    def get_embedding(self, num_units):

        self.user_emb_lookup_table = self.init_embedding_lookup_table(name="user_emb_w", total_count=self.user_count,
                                                                      embedding_dim=num_units,
                                                                      is_training=self.is_training)
        user_emb_result = tf.nn.embedding_lookup(self.user_emb_lookup_table, self.user_id)

        for key, values in self.embedding_dic.items():
            embedding_dim = values[1]
            total_count = getattr(self, key + "_count")

            if key == "item":
                self.item_emb_lookup_table = self.init_embedding_lookup_table(name="item", total_count=total_count+3,
                                                                              embedding_dim=embedding_dim,
                                                                              is_training=self.is_training)
                tf.summary.histogram('item_emb_lookup_table', self.item_emb_lookup_table)
                behavior_seq_embedding_result = tf.nn.embedding_lookup(self.item_emb_lookup_table, self.item_list)
                behavior_positive_embedding_result = tf.nn.embedding_lookup(self.item_emb_lookup_table, self.item_positive)
                # behavior_negative_embedding_result = tf.nn.embedding_lookup(self.item_emb_lookup_table, self.item_negative)

            else:
                table_name = key + "_emb_lookup_table"
                now_emb_lookup_table = self.init_embedding_lookup_table(name=key, total_count=total_count+3,
                                                                        embedding_dim=embedding_dim,
                                                                        is_training=self.is_training)
                setattr(self, table_name, now_emb_lookup_table)
                tf.summary.histogram(table_name, now_emb_lookup_table)

                #contact with item id
                now_id = getattr(self, key + "_list")
                now_emb = tf.nn.embedding_lookup(now_emb_lookup_table, now_id)
                behavior_seq_embedding_result = tf.concat([behavior_seq_embedding_result, now_emb],
                                                          axis=2, name="seq_embedding_concat")

                now_target_id = getattr(self, key + "_positive")
                now_target_emb = tf.nn.embedding_lookup(now_emb_lookup_table, now_target_id)
                behavior_positive_embedding_result = tf.concat([behavior_positive_embedding_result, now_target_emb],
                                                               axis=1,
                                                               name="pos_embedding_concat")

        behavior_seq_embedding_result_dense = tf.layers.dense(behavior_seq_embedding_result,
                                                              num_units,
                                                              activation=tf.nn.relu,
                                                              use_bias=False,
                                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
                                                              name='dense4emb')
        # (N, T_q, C)
        behavior_positive_embedding_result_dense = tf.layers.dense(behavior_positive_embedding_result, num_units,
                                                                   activation=tf.nn.relu,use_bias=False,
                                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
                                                                   name='dense4emb',reuse=True)
        # (N, T_q, C)
        behavior_seq_embedding_result_dense = self.position_embedding(behavior_seq_embedding_result_dense)

        return  behavior_seq_embedding_result, behavior_positive_embedding_result, \
                behavior_seq_embedding_result_dense, \
                behavior_positive_embedding_result_dense, \
                self.mask_index, self.item_positive, self.seq_length, user_emb_result, self.time_interval_list,\
                self.time_list, self.pos_last_list



    def tranform_list_ndarray(self,deal_data,max_len,index):

        result = np.zeros([len(self.batch_data),max_len],np.float)

        k = 0
        for t in deal_data:
            for l in range(len(t[1])):
                result[k][l] = t[k][l]
            k += 1

        return result

    def position_embedding(self, input):
        output = input
        full_position_embeddings = tf.get_variable(
            name="position_embedding_name",
            shape=[512, 128])
        position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                       [self.max_len, -1])
        num_dims = len(output.shape.as_list())
        position_broadcast_shape = []
        for _ in range(num_dims - 2):
            position_broadcast_shape.append(1)
        position_broadcast_shape.extend([self.max_len, 128])
        position_embeddings = tf.reshape(position_embeddings,
                                         position_broadcast_shape)
        output += position_embeddings

        output = tf.contrib.layers.layer_norm(
            inputs=output,
            begin_norm_axis=-1,
            begin_params_axis=-1,
            scope="lay_norm")
        output = tf.nn.dropout(output, 1.0 - 0.1)

        return output


    def concat_time_emb(self,item_seq_emb):

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

    #batch format   (user,item_list,factor_list[c,a,t],temp_index,target[item,c,a,t])
    def make_feed_dic(self, batch_data):

        def normalize_time(time):
            time = np.log(time+np.ones_like(time))
            return time/(np.mean(time)+1)

        self.batch_data = batch_data
        feed_dic = {}
        user_list = []
        time_list = []
        pos_list = []
        time = []
        self.result_item = []
        seq_length = []
        index = 0
        max_len = self.max_len
        #for target
        for key, values in self.embedding_dic.items():
            result_data_list = []
            result_data_pos_target = []
            # result_data_neg_target = []
            for onedata in self.batch_data:
                if key == "item":
                    one_list = onedata[1]
                    self.result_item.append(onedata[4][0])
                    seq_length.append(len(one_list))
                    user_list.append(onedata[0])
                    onedata[2][-1][-1] = 1
                    time_list.append(np.pad(onedata[2][-1], [0, max_len-len(onedata[2][-1])], 'constant'))
                    time.append(np.pad(onedata[5], [0, max_len-len(onedata[5])], 'constant'))
                    pos_list.append(np.pad(onedata[6], [0, max_len - len(onedata[6])], 'constant'))
                else:
                    one_list = onedata[2][index - 1]

                # max_len = int(values[3])

                #padding
                if len(one_list) < max_len:
                    one_list = np.pad(one_list, [0, max_len-len(one_list)], 'constant')

                if len(one_list) > max_len:
                    one_list = one_list[:max_len]

                result_data_list.append(one_list)
                result_data_pos_target.append(onedata[4][index])

                #neg list
                # neg_temp_list = []
                # for neg_detail in onedata[4][1]:
                #     neg_temp_list.append(neg_detail[index])
                #
                # result_data_neg_target.append(neg_temp_list)

            index = index + 1
            result_data_array = copy.deepcopy(np.array(result_data_list))
            result_data_pos_target_array = copy.deepcopy(np.array(result_data_pos_target))
            # result_data_neg_target_array = copy.deepcopy(np.array(result_data_neg_target))

            key_list = key + "_list"
            if key != "item":
                now_placeholder = getattr(self, key_list)
            else:
                now_placeholder = self.item_list

            feed_dic[now_placeholder] = result_data_array

            key_pos_target_id = key + "_positive"
            now_placeholder_target = getattr(self, key_pos_target_id)
            feed_dic[now_placeholder_target] = result_data_pos_target_array

            # key_neg_target_id = key + "_negative"
            # now_placeholder_target = getattr(self, key_neg_target_id)
            # feed_dic[now_placeholder_target] = result_data_neg_target_array

        label_list = np.array([[i[3] for i in batch_data]]).T

        #y placeholder
        feed_dic[self.mask_index] = label_list
        feed_dic[self.seq_length] = np.array(seq_length)
        feed_dic[self.user_id] = np.array(user_list)
        feed_dic[self.time_interval_list] = np.array(time_list)
        feed_dic[self.time_list] = np.array(time)
        feed_dic[self.pos_last_list] = np.array(pos_list)

        return feed_dic






