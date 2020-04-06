from Embedding.base_embedding import Base_embedding
import tensorflow as tf

class Bprmf_embedding(Base_embedding):

    def __init__(self,is_training=True,config_file = None,user_count = 100,item_count = 100):

        super(Bprmf_embedding, self).__init__(is_training,config_file)
        self.user_count = user_count
        #give a large number
        self.item_count = item_count + 9999
        self.hidden_dim = 60
        self.is_training = is_training
        self.init_placeholders()


    def init_placeholders(self):

        # [B] user id
        self.user_id = tf.placeholder(tf.int32, [None, ])

        # [B] item postive target id
        self.item_pos_target_id = tf.placeholder(tf.int32, [None, ])

        # [B] item negtive target id
        self.item_neg_target_id = tf.placeholder(tf.int32, [None, ])

    def get_embedding(self):

        self.user_emb_lookup_table = self.init_embedding_lookup_table(name="user_emb_w", total_count=self.user_count,
                                                                      embedding_dim=self.hidden_dim,
                                                                      is_training=self.is_training)

        self.item_emb_lookup_table = self.init_embedding_lookup_table(name="item_emb_w", total_count=self.item_count,
                                                                      embedding_dim=self.hidden_dim,
                                                                      is_training=self.is_training)


        self.item_b_lookup_table   = self.init_embedding_lookup_table(name="item_b", total_count=self.item_count,
                                                                      embedding_dim=1, is_training=self.is_training)

        u_embedding_result = tf.nn.embedding_lookup(self.user_emb_lookup_table, self.user_id)
        item_pos_embedding_result = tf.nn.embedding_lookup(self.item_emb_lookup_table, self.item_pos_target_id)
        item_neg_embedding_result = tf.nn.embedding_lookup(self.item_emb_lookup_table, self.item_neg_target_id)

        item_pos_b = tf.nn.embedding_lookup(self.item_b_lookup_table, self.item_pos_target_id)
        item_neg_b = tf.nn.embedding_lookup(self.item_b_lookup_table, self.item_neg_target_id)

        return u_embedding_result, item_pos_embedding_result, item_neg_embedding_result, item_pos_b, item_neg_b

    def make_feed_dic(self, batch_data):
        self.result_item = []
        self.batch_data = batch_data
        feed_dict = {}

        user_id_list = []
        item_pos_list = []
        item_neg_list = []
        for one_data in self.batch_data:
            user_id_list.append(one_data[0])
            item_pos_list.append(one_data[1])
            item_neg_list.append(one_data[2])
            self.result_item.append(one_data[1])

        feed_dict[self.user_id] = user_id_list
        feed_dict[self.item_pos_target_id] = item_pos_list
        feed_dict[self.item_neg_target_id] = item_neg_list
        return feed_dict

    # def init_embedding_lookup_table(self,name,total_count,embedding_dim,is_training=True):
    #     lookup_table = super(Bprmf_embedding, self).init_embedding_lookup_table(name, total_count,
    #                                                                             embedding_dim,
    #                                                                             is_training=True)
    #     return lookup_table
    #
    #
