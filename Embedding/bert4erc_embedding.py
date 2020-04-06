import tensorflow as tf
import numpy as np
from Embedding.base_embedding import Base_embedding


class Bert2erc_embedding(Base_embedding):

    def __init__(self,is_training = True, item_count=100):

        super(Bert2erc_embedding, self).__init__(self)
        self.item_count = item_count
        self.init_placeholders()

    def init_placeholders(self):

        with tf.variable_scope("input_layer"):

            # [B] user id
            self.user_id = tf.placeholder(tf.int32, [None, ], name="user")
            # [B] item id
            self.item_list = tf.placeholder(tf.int32, [None, None], name="item_seq")
            # [B] masked item id
            self.masked_item = tf.placeholder(tf.int32, [None, ], name="masked_item")
            # [B] item label
            self.y = tf.placeholder(tf.float32, [None, 1], name="label")

    def get_embedding(self, num_units):

        self.item_emb = tf.get_variable("item_emb", [self.item_count, self.item_count])

        position_emb = tf.one_hot(self.item_list, self.item_count, dtype=tf.float32)
        behavior_seq_emb = tf.nn.embedding_lookup(self.item_emb, self.item_list) + position_emb

        positon_target_emb = tf.one_hot(self.masked_item, self.item_count, dtype=tf.float32)
        behavior_target_emb = tf.nn.embedding_lookup(self.item_emb, self.masked_item) + positon_target_emb

        behavior_target_embedding_dence = tf.layers.dense(behavior_target_emb, num_units, activation=tf.nn.relu,
                                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
                                                       name='dense4emb')
        behavior_seq_embedding_dence = tf.layers.dense(behavior_seq_emb, num_units, activation=tf.nn.relu,
                                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
                                                       name='dense4emb')
        return behavior_seq_embedding_dence, behavior_target_embedding_dence, self.y, self.item_emb

    def make_feed_dic(self, batch_data):

        self.batch_data = batch_data
        feed_dic = {}

        item_list = []
        mask_item_list = []
        label_list = []
        user_list = []
        item_list_len = []
        for one_data in self.batch_data:
            item_list_len.append(len(one_data[1]))

        max_len = max(item_list_len) + 1

        for one_data in self.batch_data:
            user_list.append(one_data[0])
            one_list = np.array(one_data[1] + np.array(one_data[2]))

            # padding
            if len(one_list) < max_len:
                padding_list = [0] * (max_len - len(one_list))
                item_list = one_list + padding_list

            if len(one_list) > max_len:
                one_list = one_list.subList(0, max_len)

            item_list.append(one_list)
            mask_item_list.append(one_data[2])
            label_list.append(one_data[3])

        label_list = np.array(label_list).T

        feed_dic[self.user_id] = user_list
        feed_dic[self.item_list] = item_list
        feed_dic[self.masked_item] = mask_item_list
        feed_dic[self.y] = label_list

        return feed_dic
















