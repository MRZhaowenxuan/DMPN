import tensorflow as tf
from Embedding.bprmf_embedding import Bprmf_embedding
from Model.base_model import base_model
import math
class Bprmf_model(base_model):

    def __init__(self, FLAGS,Embeding,sess):

        super(Bprmf_model, self).__init__(FLAGS,Embeding)
        self.now_bacth_data_size = tf.placeholder(tf.int32, shape=[], name="batch_size")
        # Summary Writer
        self.learning_rate = tf.placeholder(tf.float64, [], name="learning_rate")
        self.build_model()
        self.init_variables(sess,self.checkpoint_path_dir)


    def build_model(self):
        user_emb, item_pos_emb, item_neg_emb, \
        item_pos_b, item_neg_b = self.embedding.get_embedding()

        x = item_pos_b - item_neg_b + tf.reduce_sum(tf.multiply(user_emb, (item_pos_emb - item_neg_emb)), 1)
        self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))

        l2_norm = tf.add_n([
            tf.nn.l2_loss(user_emb),
            tf.nn.l2_loss(item_pos_emb),
            tf.nn.l2_loss(item_neg_emb)
        ])
        regulation_rate = 5e-5
        self.bprloss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(x)))
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.bprloss)
        self.predict_behavior_emb = user_emb

        with tf.name_scope('Bpr_expriment'):
            tf.summary.scalar('mfauc',self.mf_auc)
            tf.summary.scalar('Loss' ,self.bprloss)
            tf.summary.scalar('l2_norm', l2_norm)
            tf.summary.scalar('Learning_rate',self.learning_rate)

        self.summery()

    def train(self, sess, batch_data, learning_rate,add_summary=False,global_step=0):

        input_feed = self.embedding.make_feed_dic(batch_data)
        input_feed[self.learning_rate] = learning_rate
        sess.run(self.train_op, input_feed)
        self.loss, merged = sess.run([self.bprloss,self.merged], input_feed)
        return self.loss,merged

    def metrics(self,sess,batch_data,global_step,name="Sun"):
        input_dic = self.embedding.make_feed_dic(batch_data=batch_data)
        auc = sess.run(self.mf_auc,input_dic)
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=auc)])
        self.eval_writer.add_summary(summary,global_step=global_step)
        return auc

    def metrics_topK(self, sess, batch_data, global_step, topk):

        input_dic = self.embedding.make_feed_dic(batch_data=batch_data)
        # input_dic[self.now_bacth_data_size] = len(batch_data)
        item_lookup_table_T = tf.transpose(self.embedding.item_emb_lookup_table)
        item_result = tf.matmul(self.predict_behavior_emb, item_lookup_table_T)

        # get topK index
        indices = tf.nn.top_k(item_result, topk).indices
        indices_result = sess.run(indices, input_dic)
        # indices_result, item_result = sess.run([indices, item_result], input_dic)
        result_item = self.embedding.result_item
        total_count = 0
        recall_count = 0
        ndcg_value_list = []
        for one_user_data in indices_result:
            one_user_data = list(one_user_data)
            if result_item[total_count] in one_user_data:
                recall_count = recall_count + 1

            for i in range(len(one_user_data)):
                if result_item[total_count] == one_user_data[i]:
                    ndcg_value = math.log(2) / math.log(i + 2)
                    ndcg_value_list.append(ndcg_value)
                    break
            total_count = total_count + 1

        recall_rate = recall_count / total_count

        # the default ndcg value is 0
        if len(ndcg_value_list) > 0:
            avg_ndcg = float(sum(ndcg_value_list)) / len(batch_data)
        else:
            avg_ndcg = 0

        summary_recall_rate = tf.Summary(value=[tf.Summary.Value(tag="recall_rate", simple_value=recall_rate)])
        self.train_writer.add_summary(summary_recall_rate, global_step=global_step)
        summary_avg_ndcg = tf.Summary(value=[tf.Summary.Value(tag="avg_ndcg", simple_value=avg_ndcg)])
        self.train_writer.add_summary(summary_avg_ndcg, global_step=global_step)
        return recall_rate, avg_ndcg
