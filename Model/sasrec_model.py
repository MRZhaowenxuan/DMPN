import tensorflow as tf
from Model.Modules.gru import GRU
from Model.Modules.multihead_attention import Attention
from Model.Modules.net_utils import gather_indexes,layer_norm
from Model.base_model import base_model


class SASRec_model(base_model):

    def __init__(self, FLAGS, Embedding, sess):
        super(SASRec_model, self).__init__(FLAGS, Embedding)
        self.now_bacth_data_size = tf.placeholder(tf.int32, shape=[], name="batch_size")
        self.build_model()
        self.init_variables(sess, self.checkpoint_path_dir)

    def build_model(self):
        num_units = self.FLAGS.num_units
        num_heads = self.FLAGS.num_heads
        num_blocks = self.FLAGS.num_blocks
        dropout_rate = self.FLAGS.dropout

        attention_net = Attention()
        gru_net_ins = GRU()

        self.sequence_embedding, self.positive_embedding, self.negative_embedding, \
        self.behavior_embedding_result_dense, self.positive_embedding_result_dense, \
        self.negative_embedding_result_dense, self.mask_index, self.label_ids, \
        self.seq_length = self.embedding.get_embedding(num_units)

        with tf.variable_scope('ShortTermIntentEncoder'):

            long_term_intent_temp = attention_net.self_attention_single(enc=self.behavior_embedding_result_dense,
                                                                 num_units=128, num_heads=num_heads,
                                                                 num_blocks=num_blocks,
                                                                 dropout_rate=dropout_rate, is_training=True,
                                                                 reuse=False)
            self.long_term_intent = gather_indexes(batch_size=self.now_bacth_data_size, seq_length=self.FLAGS.max_len,
                                                   width=self.FLAGS.num_units, sequence_tensor=long_term_intent_temp,
                                                   positions=self.mask_index)
        with tf.variable_scope("OutputLayer"):
            self.predict_behavior_emb = self.long_term_intent

            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)

            self.mf_auc = tf.reduce_mean(tf.to_float((tf.reduce_sum(tf.multiply(tf.expand_dims(self.predict_behavior_emb, 1),
                                                                                tf.expand_dims(self.positive_embedding_result_dense, 1) - self.negative_embedding_result_dense), 2)) > 0))


            l2_norm = tf.add_n([
                tf.nn.l2_loss(self.sequence_embedding),
                tf.nn.l2_loss(self.positive_embedding),
                tf.nn.l2_loss(self.negative_embedding)
            ])
            regulation_rate = self.FLAGS.regulation_rate

            item_lookup_table_T = tf.transpose(self.embedding.item_emb_lookup_table)
            logits = tf.matmul(self.predict_behavior_emb, item_lookup_table_T)
            log_probs = tf.nn.log_softmax(logits)
            label_ids = tf.reshape(self.label_ids, [-1])
            one_hot_labels = tf.one_hot(label_ids, depth=500000, dtype=tf.float32)
            self.loss_origin = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            lstur_loss = regulation_rate * l2_norm + tf.reduce_mean(self.loss_origin)

        with tf.name_scope("LearningtoRankLoss"):
            self.loss = lstur_loss
            tf.summary.scalar("l2_norm", l2_norm)
            tf.summary.scalar("Training Loss", self.loss)
            tf.summary.scalar("Learning_rate", self.learning_rate)

        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)

        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.FLAGS.max_gradient_norm)

        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params))

        self.summery()

    def train(self, sess, batch_data, learning_rate, add_summary=False, global_step=0):
        self.init_reserved_field(sess)
        input_dict = self.embedding.make_feed_dic(batch_data)
        input_dict[self.learning_rate] = learning_rate
        input_dict[self.now_bacth_data_size] = len(batch_data)
        output_feed = [self.loss, self.merged, self.train_op]
        outputs = sess.run(output_feed, input_dict)

        return outputs[0], outputs[1]

    def inference(self, sess, batch_data):
        input_dict = self.embedding.make_feed_dic(batch_data)
        output = sess.run(self.y_postive, self.y_negative, self.mf_auc, input_dict)
        return output

    def metrics(self, sess, batch_data, global_step, name):
        self.init_reserved_field(sess)
        input_dict = self.embedding.make_feed_dic(batch_data)
        input_dict[self.now_bacth_data_size] = len(batch_data)
        auc = sess.run(self.mf_auc, input_dict)
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=auc)])
        self.train_writer.add_summary(summary, global_step)

        return auc