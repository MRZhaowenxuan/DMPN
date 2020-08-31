import tensorflow as tf

from Model.Modules.gru import GRU
from Model.Modules.multihead_attention import Attention
from Model.Modules.net_utils import gather_indexes,layer_norm
from Model.base_model import base_model


class DFM_model(base_model):

    def __init__(self, FLAGS, Embedding, sess):
        super(DFM_model, self).__init__(FLAGS, Embedding)
        self.now_bacth_data_size = tf.placeholder(tf.int32, shape=[], name="batch_size")
        self.build_model()
        self.init_variables(sess, self.checkpoint_path_dir)

    def build_model(self):
        num_units = self.FLAGS.num_units
        num_heads = self.FLAGS.num_heads
        num_blocks = self.FLAGS.num_blocks
        dropout_rate = self.FLAGS.dropout
        embedding_size = self.FLAGS.itemid_embedding_size
        use_mmoe = self.FLAGS.use_mmoe

        attention_net = Attention()
        gru_net_ins = GRU()

        self.sequence_embedding, self.positive_embedding, \
        self.behavior_embedding_result_dense, self.positive_embedding_result_dense, \
        self.mask_index, self.label_ids, \
        self.seq_length, user_embedding, self.time_interval, \
        self.time, self.pos_last_list = self.embedding.get_embedding(num_units)

        with tf.variable_scope("LongTermIntentEncoder"):
            attention_output = attention_net.self_attention(enc=self.behavior_embedding_result_dense,
                                                            num_units=128, num_heads=num_heads, num_blocks=num_blocks,
                                                            dropout_rate=dropout_rate, is_training=True, reuse=False,
                                                            key_length=self.seq_length, query_length=self.seq_length)
            attention_pooling = tf.reduce_sum(attention_output, 1)
            # flat_attention_output = tf.reshape(attention_output, shape=[-1, embedding_size])
            # flat_behavior_emb = tf.reshape(self.behavior_embedding_result_dense, shape=[-1, embedding_size])
            # concat_emb = tf.concat([flat_attention_output, flat_behavior_emb], axis=1)
            # net = tf.layers.dense(concat_emb, embedding_size, activation=tf.nn.relu, use_bias=False)
            # att_wgt = tf.layers.dense(net, 1, activation=tf.nn.relu, use_bias=False)
            # att_wgt = tf.reshape(att_wgt, shape=[-1, seq_len])
            # att_wgt = att_wgt / (embedding_size ** 0.5)
            # att_wgt = tf.nn.softmax(att_wgt)
            # att_wgt = tf.reshape(att_wgt, shape=[-1, seq_len, 1])
            # output = tf.multiply(attention_output, att_wgt)
            # attention_pooling = tf.reduce_sum(output, 1)

        with tf.variable_scope("EnhanceUserPreferenceIntentEncoder"):
            gru_input = tf.concat([self.behavior_embedding_result_dense,
                                   tf.expand_dims(self.time_interval, 2),
                                   tf.expand_dims(self.pos_last_list, 2)], 2)

            self.gru_output = gru_net_ins.time_aware_gru_net(hidden_units=num_units,
                                                             input_data=gru_input,
                                                             input_length=tf.add(self.seq_length, -1))

            self.gru_output = gather_indexes(batch_size=self.now_bacth_data_size,
                                             seq_length=self.FLAGS.max_len,
                                             width=self.FLAGS.num_units,
                                             sequence_tensor=self.gru_output,
                                             positions=tf.add(self.mask_index, -1))
            _, seq_len, size = self.behavior_embedding_result_dense.get_shape().as_list()
            feature_emb = tf.reshape(self.behavior_embedding_result_dense, [-1, seq_len * size])
            concat_output = tf.concat([self.gru_output, attention_pooling, feature_emb], axis=1)
            dence1 = tf.layers.dense(concat_output, concat_output.get_shape().as_list()[1] // 2, activation=tf.nn.relu, use_bias=False)
            self.user_preference = tf.layers.dense(dence1, num_units, activation=tf.nn.relu, use_bias=False)

        if use_mmoe:
            print("mmoe")
            with tf.variable_scope("mmoe"):
                num_expert = 8
                expert_outputs = []
                for _ in range(num_expert):
                    expert_output = tf.layers.dense(self.user_preference, embedding_size, activation=tf.nn.relu, use_bias=False)
                    expert_output = tf.expand_dims(expert_output, axis=2)  # [B, 64, 1]
                    expert_outputs.append(expert_output)
                expert_outputs = tf.concat(expert_outputs, axis=2)  # [B, 64, 8]

                gate_network = tf.layers.dense(self.user_preference, num_expert, activation=tf.nn.softmax, use_bias=False)
                gate_network_dim = tf.expand_dims(gate_network, axis=1)  # [B, 1, 8]
                weighted_expert_ouptputs = tf.tile(gate_network_dim, [1, embedding_size, 1]) * expert_outputs
                final_output = tf.reduce_sum(weighted_expert_ouptputs, axis=2)
                self.user_preference = tf.layers.dense(final_output, num_units, activation=tf.nn.relu, use_bias=False)

        with tf.variable_scope("OutputLayer"):
            self.predict_behavior_emb = layer_norm(self.user_preference)
            l2_norm = tf.add_n([
                tf.nn.l2_loss(self.sequence_embedding),
                tf.nn.l2_loss(self.positive_embedding),
                tf.nn.l2_loss(user_embedding)
            ])
            regulation_rate = self.FLAGS.regulation_rate

            item_lookup_table_T = tf.transpose(self.embedding.item_emb_lookup_table)
            #print("item_embedding:", item_lookup_table_T.get_shape().as_list())
            logits = tf.matmul(self.predict_behavior_emb, item_lookup_table_T)
            log_probs = tf.nn.log_softmax(logits)
            label_ids = tf.reshape(self.label_ids, [-1])
            one_hot_labels = tf.one_hot(label_ids, depth=self.embedding.item_count+3, dtype=tf.float32)
            self.loss_origin = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            lstur_loss = regulation_rate * l2_norm + tf.reduce_mean(self.loss_origin)

        with tf.name_scope("LearningtoRankLoss"):
            self.loss = lstur_loss
            if self.FLAGS.add_summary:
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