import tensorflow as tf
from Model.Modules.multihead_attention import Attention
from Model.Modules.net_utils import gather_indexes
from Model.base_model import base_model
class Atrank_model(base_model):
    def __init__(self, FLAGS,Embeding,sess):

        super(Atrank_model, self).__init__(FLAGS,Embeding) # 此处修改了
        self.now_bacth_data_size = tf.placeholder(tf.int32, shape=[],name='batch_size')

        self.build_model()
        # parm_list = [n.name for n in tf.get_default_graph().as_graph_def().node]
        # print(parm_list)
        self.init_variables(sess,self.checkpoint_path_dir)


    def build_model(self):


        num_units = self.FLAGS.num_units
        num_heads = self.FLAGS.num_heads
        num_blocks = self.FLAGS.num_blocks
        dropout_rate = self.FLAGS.dropout

        sequence_embedding, positive_embedding, negative_embedding, \
        self.behavior_embedding_result, self.positive_embedding_result,\
        self.negative_embedding_result, self.mask_index,self.label_ids,\
        self.seq_length = self.embedding.get_embedding(num_units)


        #print('Temporal CNN lay')
        #with tf.variable_scope("TemporalCNN"):
            #compressed_behavior_embedding= tf.layers.conv1d(inputs = self.behavior_embedding_result, filters=128,kernel_size=[10],strides=5,activation='relu')
            #compressed_behavior_embedding = tf.layers.average_pooling1d(inputs = compressed_behavior_embedding, pool_size= 5,strides=2 )
        enc = self.behavior_embedding_result

        attention_net = Attention()
        # print('behavior_embedding_result')
        # print(self.behavior_embedding_result.shape.as_list())
        with tf.variable_scope("UserHistoryEncoder"):
            enc = attention_net.self_attention(enc=enc, num_units=128,
                                                        num_heads=num_heads, num_blocks = num_blocks,
                                                        dropout_rate= dropout_rate, is_training=True, reuse=False,
                                                        key_length=self.seq_length, query_length = self.seq_length)

            self.user_h = enc

        #self.logits = tf.layers.dense(inputs=attention_result, units=1,kernel_regularizer=tf.contrib.layers.l2_regularizer(self.FLAGS.regulation_rate))

        #self.logits= tf.reduce_sum(tf.multiply(attention_result,self.item_target_embedding_result),1,keep_dims=True)
        with tf.variable_scope('UserHistoryDecoder'):
            positive_embedding_result = tf.expand_dims(self.positive_embedding_result,1)
            attention_result = attention_net.vanilla_attention(self.user_h, positive_embedding_result, num_units,
                                                               num_heads, num_blocks, dropout_rate, is_training=True,
                                                               reuse=False, key_length=self.seq_length,
                                                               query_length=tf.ones_like(
                                                                   positive_embedding_result[:, 0, 0], dtype=tf.int32))
            self.predict_behavior_emb = attention_result

            self.logits_positive = tf.reduce_sum(tf.multiply(attention_result, self.positive_embedding_result), 1)
            self.logits_negative = tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.expand_dims(attention_result,1),
                                                             self.negative_embedding_result), 2),1)

            #loss = tf.reduce_sum(tf.multiply(predict_behavior_emb,(self.positive_embedding_result - self.negative_embedding_result)), 1)
            #self.mf_auc = tf.reduce_sum(tf.multiply(tf.expand_dims(self.predict_behavior_emb,1),
                                                   #(tf.expand_dims(self.positive_embedding_result,1) -self.negative_embedding_result)),1)
            #self.mf_auc = tf.reduce_mean(tf.to_float(self.mf_auc > 0))

            mf_auc = tf.expand_dims(self.positive_embedding_result, 1) - self.negative_embedding_result
            mf_auc = tf.multiply(tf.expand_dims(self.predict_behavior_emb, 1), mf_auc)
            mf_auc = tf.reduce_sum(mf_auc, 2)
            mf_auc = tf.to_float(mf_auc > 0)

            # loss = tf.reduce_sum(tf.multiply(predict_behavior_emb,(self.positive_embedding_result - self.negative_embedding_result)), 1)
            self.mf_auc = tf.reduce_mean(mf_auc)

            l2_norm = tf.add_n([
                tf.nn.l2_loss(sequence_embedding),
                tf.nn.l2_loss(positive_embedding),
                tf.nn.l2_loss(negative_embedding)
            ])
            regulation_rate = self.FLAGS.regulation_rate
            #bsbeloss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(self.logits)))
            loss1 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_positive,
                    labels=tf.ones_like(self.logits_positive, dtype=tf.float32))
                )
            loss2 = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits_negative,
                    labels=tf.zeros_like(self.logits_negative, dtype=tf.float32))
                )
            self.loss = (loss1+loss2)/2+regulation_rate*l2_norm



        # ============== Eval ===============
            positive_logit = tf.reduce_sum(tf.multiply(attention_result,self.positive_embedding_result),1)
            negative_logit = tf.reduce_sum(tf.multiply(attention_result,self.negative_embedding_result),1)
            self.y_positive = tf.sigmoid(positive_logit)
            self.y_negative = tf.sigmoid(negative_logit)



        #l2_norm = tf.add_n([tf.nn.l2_loss(sequence_embedding),tf.nn.l2_loss(behavior_embedding)])
        #l2_norm = tf.add_n([tf.nn.l2_loss(behavior_embedding)])
        #reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-5), tf.trainable_variables())


        with tf.name_scope('LearningtoRankLoss'):
            #self.loss = bsbeloss
            #sigmoid_loss = tf.reduce_mean(tf.sigmoid(self.logits))
            tf.summary.scalar('mfauc',self.mf_auc)
            #tf.summary.scalar('Loss', sigmoid_loss)
            tf.summary.scalar('l1_norm', l2_norm)
            tf.summary.scalar('Training Loss', self.loss)
            tf.summary.scalar('Learning_rate',self.learning_rate)


        trainable_params = tf.trainable_variables()

        # Compute gradients of loss w.r.t. all trainable variables
        gradients = tf.gradients(self.loss, trainable_params)

        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.FLAGS.max_gradient_norm)

        # Update the model
        self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params))

        self.summery()


        # self.metrics(global_step=self.FLAGS.global_step)
        # Step variable靠
        # TODO: 这里原来是0
        # self.global_step          = tf.Variable(0, trainable=False, name='global_step')
        # self.global_epoch_step    = tf.Variable(0, trainable=False, name='global_epoch_step')
        # self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)


        #tf.summary.histogram('embedding/1_item_emb', self.behavior_embedding)
        # tf.summary.histogram('embedding/2_cate_emb', cate_emb_w)
        # tf.summary.histogram('embedding/4_final', h_emb)
        # tf.summary.histogram('attention_output', u_emb)

    #TODO:add_summary=False
    def train(self, sess, batch_data, learning_rate,add_summary=False,global_step = 0):
        #set reverse field 0
        self.init_reserved_field(sess)
        input_dic = self.embedding.make_feed_dic(batch_data=batch_data)
        input_dic[self.learning_rate] = learning_rate
        input_dic[self.now_bacth_data_size] = len(batch_data)

        output_feed = [self.loss, self.merged, self.train_op]
        outputs = sess.run(output_feed, input_dic)
        return outputs[0],outputs[1]

    def inference(self, sess,batch_data):

        input_dic = self.embedding.make_feed_dic(batch_data=batch_data)
        output = sess.run(self.y_positive,self.y_negative, self.mf_auc, input_dic)
        return output


    #over—write
    #the error named "AUC"
    def metrics(self,sess,batch_data,global_step,name):
        input_dic = self.embedding.make_feed_dic(batch_data=batch_data)
        input_dic[self.now_bacth_data_size] = len(batch_data)
        #[loss,loss1,loss2,loss3]= sess.run([self.loss,self.loss1,self.loss2,self.loss3],input_dic)
        #print(loss)
        #print(loss1)
        #print(loss2)
        #print(loss3)

        auc = sess.run(self.mf_auc,input_dic)

        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=auc)])
        self.train_writer.add_summary(summary, global_step=global_step)
        return auc