import json
import numpy as np
import tensorflow as tf
import os
from Model.Modules.net_utils import *
from sklearn.metrics import roc_auc_score
from Embedding.base_embedding import Base_embedding
from util.model_log import create_log
import time
import math
'''
Base class of embedding module for all models.
'''


class base_model(object):


    def __init__(self,FLAGS,Embedding):

        self.FLAGS = FLAGS
        self.version = self.FLAGS.version
        self.learning_rate = tf.placeholder(tf.float64, [], name = "learning_rate")


        #self.embedding = Embedding



        #set the defalut check point path
        if self.FLAGS.checkpoint_path_dir != None:
            self.checkpoint_path_dir = self.FLAGS.checkpoint_path_dir

        else:
            self.checkpoint_path_dir = "data/check_point/" + self.FLAGS.type + "_" + self.FLAGS.experiment_type + "_" + self.version
            if not os.path.exists(self.checkpoint_path_dir):
                os.makedirs(self.checkpoint_path_dir)


        self.init_optimizer()
        self.embedding = Embedding
        log_ins = create_log()
        self.logger = log_ins.logger


    def init_variables(self,sess,path,var_list=None):

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if(self.FLAGS.load_type == "full"):
            self.restore(sess,path=path)

        elif(self.FLAGS.load_type == "fine_tune"):
            #load graph get the last meta
            path = self.FLAGS.fine_tune_load_path
            meta_list = os.listdir(path)
            for i in meta_list:
                # os.path.splitext():分离文件名与扩展名
                if os.path.splitext(i)[1] == '.meta':
                    meta_file = i

            meta_file_path = os.path.join(path,meta_file)
            self.restore(sess,path=path,variable_list=var_list)

        elif(self.FLAGS.load_type == "from_scratch"):
            pass

    '''Get the Optimizer by flag

    Args:


    Returns:
      optimizer'''

    def init_optimizer(self):
        # Gradients and SGD update operation for training the model
        if self.FLAGS.optimizer == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self.FLAGS.optimizer =='adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.FLAGS.optimizer == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)



    def build_model(self):
        pass

    '''The train process abstract class

     Args:
                
        sess: tensorflow sess
        bathch_data: the batch data for embedding
        add_summary:  whether to add summary

     Returns:
       loss and train op'''
    def train(self, sess, batch_data, learning_rate, add_summary=False, global_step=0):
        pass


    def inference(self, sess,batch_data):
        pass



    #init reserved field table 0
    #to look item lookup table
    def init_reserved_field(self,sess):
        # input_dic = self.embedding.make_feed_dic(batch_data=batch_data)
        for key, values in self.embedding.embedding_dic.items():
            table_name = key + "_emb_lookup_table"
            now_emb_lookup_table = getattr(self.embedding, table_name)
            embedding_size = int(values[1])
            assign_zero_op = tf.assign(now_emb_lookup_table[0], [0] * embedding_size)
            sess.run(assign_zero_op)







    def save(self,sess,global_step = None,path=None,variable_list=None):
        #print(self.global_step)

        if path == None:
            path = self.checkpoint_path_dir
        # if not os.path.exists(path):
        #     os.makedirs(path)

        path = os.path.join(path, "model.ckpt")

        saver = tf.train.Saver(var_list = variable_list)
        save_path = saver.save(sess, save_path = path, global_step=global_step)


        self.logger.info('model saved at %s' % save_path)

    def restore(self,sess,path,variable_list=None,graph_path=None):
        if graph_path != None:
            saver = tf.train.import_meta_graph(graph_path)
        else:
            saver = tf.train.Saver(var_list=variable_list)

        saver.restore(sess, tf.train.latest_checkpoint(path))
        self.logger.info('model restored from %s' % path)


    def metrics(self,sess,batch_data,global_step,name):
        #set reverse field 0
        self.init_reserved_field(sess)

        input_dic = self.embedding.make_feed_dic(batch_data=batch_data)
        y_hat,y = sess.run([self.y_hat, self.y],input_dic)
        y_hat = y_hat.flatten()
        y = y.flatten()
        auc = roc_auc_score(y,y_hat)
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=auc)])
        self.train_writer.add_summary(summary, global_step=global_step)
        return auc

    '''
    according to model generate new next item
    to calculate recall rate or ndcg value
    '''
    def metrics_topK(self,sess,batch_data,global_step,topk):
        #set reverse field 0
        # self.init_reserved_field(sess)

        input_dic = self.embedding.make_feed_dic(batch_data=batch_data)
        input_dic[self.now_bacth_data_size] = len(batch_data)
        item_lookup_table_T = tf.transpose(self.embedding.item_emb_lookup_table)
        item_result = tf.matmul(self.predict_behavior_emb, item_lookup_table_T)
        #print(item_result)

        #get topK index
        indices = tf.nn.top_k(item_result,  topk).indices
        indices_result, item_result = sess.run([indices, item_result], input_dic)
        result_item = self.embedding.result_item
        #print(result_item)
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

        #the default ndcg value is 0
        if len(ndcg_value_list) > 0:
            avg_ndcg = float(sum(ndcg_value_list))/len(batch_data)
        else:
            avg_ndcg = 0

        recall_rate_tag = "recall_rate@" + str(topk)
        summary_recall_rate = tf.Summary(value=[tf.Summary.Value(tag=recall_rate_tag, simple_value=recall_rate)])
        self.train_writer.add_summary(summary_recall_rate, global_step=global_step)
        avg_ndcg_tag = "avg_ndcg@" + str(topk)
        summary_avg_ndcg = tf.Summary(value=[tf.Summary.Value(tag=avg_ndcg_tag, simple_value=avg_ndcg)])
        self.train_writer.add_summary(summary_avg_ndcg, global_step=global_step)

        return recall_rate, avg_ndcg

    def test(self, sess, batch_data, global_step):
        hr_5, ndcg_5 = self.metrics_topK(sess, batch_data, global_step, 5)
        hr_10, ndcg_10 = self.metrics_topK(sess, batch_data, global_step, 10)
        hr_20, ndcg_20 = self.metrics_topK(sess, batch_data, global_step, 20)
        return hr_5, ndcg_5, hr_10, ndcg_10, hr_20, ndcg_20




    def summery(self):

        self.merged = tf.summary.merge_all()
        timeArray = time.localtime(time.time())
        timeStr = time.strftime("%Y-%m-%d--%H:%M:%S", timeArray)
        filename = "data/tensorboard_result/"
        type = self.FLAGS.type
        experiment_type = self.FLAGS.experiment_type
        version = self.FLAGS.version

        filename = filename + type + "_" + experiment_type + "_" + version + "_" + self.FLAGS.lstur_type + "_" + timeStr
        # Summary Writer
        self.train_writer = tf.summary.FileWriter(filename  + '/tensorboard_train')
        self.eval_writer  = tf.summary.FileWriter(filename + '/tensorboard_eval')
