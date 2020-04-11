import time
import random
import numpy as np
import tensorflow as tf
from Embedding.behavior_embedding_nodec import Behavior_embedding_nodec
from Model.dib_model import DIB_model
from util.model_log import create_log
from DataHandle.get_input_data import DataInput
from Model.bprmf_model import Bprmf_model
from Model.atrank_model import Atrank_model
from Model.slirec_model import SLiRec_model
from Model.sasrec_model import SASRec_model
from Model.gru4rec_model import GRU4Rec_model
from Model.lstur_model import LSTUR_model
from Model.bert4rec_model import BERT4Rec_model
from Model.dmpn_model import DMPN_model
from DataHandle.get_origin_data import Get_origin_data
from Embedding.bprmf_embedding import Bprmf_embedding
from Embedding.lstur_embedding import Lstur_embedding
from Embedding.no_embedding import No_embedding
from Embedding.postion_emb import Position_embedding
from config.model_parameter import model_parameter
from Prepare.prepare_data_behavior import prepare_data_behavior
from Prepare.prepare_data_bpr import prepare_data_bpr
import os

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)



class Train_main_process:

    def __init__(self):

        start_time = time.time()
        model_parameter_ins = model_parameter()
        experiment_name = model_parameter_ins.flags.FLAGS.experiment_name
        self.FLAGS = model_parameter_ins.get_parameter(experiment_name).FLAGS

        log_ins = create_log(type=self.FLAGS.type, experiment_type=self.FLAGS.experiment_type,
                             version=self.FLAGS.version)
        self.logger = log_ins.logger
        self.logger.info("hello world the experiment begin")

        # logger.info("The model parameter is :" + str(self.FLAGS._parse_flags()))

        #init data and embeding
        get_origin_data_ins = Get_origin_data(FLAGS=self.FLAGS)
        if self.FLAGS.experiment_type == "dib" \
                or self.FLAGS.experiment_type == "no_emb" \
                or self.FLAGS.experiment_type == "slirec" \
                or self.FLAGS.experiment_type == "lstur" \
                or self.FLAGS.experiment_type == "sasrec" \
                or self.FLAGS.experiment_type == "grurec" \
                or self.FLAGS.experiment_type == "bert" \
                or self.FLAGS.experiment_type == "dmpn" \
                or self.FLAGS.experiment_type == "atrank":

            prepare_data_behavior_ins = prepare_data_behavior(self.FLAGS, get_origin_data_ins.origin_data)

        elif self.FLAGS.experiment_type == "bpr":
            prepare_data_behavior_ins = prepare_data_bpr(self.FLAGS, get_origin_data_ins.origin_data)


        self.logger.info('DataHandle Process.\tCost time: %.2fs' % (time.time() - start_time))
        start_time = time.time()

        #embedding
        if self.FLAGS.experiment_type == "no_emb":
            config_file = "config/no_embedding__dic.csv"
            self.emb = No_embedding(self.FLAGS.is_training, config_file)

        elif self.FLAGS.experiment_type == "bpr":
            self.emb = Bprmf_embedding(self.FLAGS.is_training,self.FLAGS.embedding_config_file,
                                       prepare_data_behavior_ins.user_count,
                                       prepare_data_behavior_ins.item_count)

        else:
            self.emb = Lstur_embedding(self.FLAGS.is_training, self.FLAGS.embedding_config_file,
                                       prepare_data_behavior_ins.user_count,
                                       prepare_data_behavior_ins.item_count,
                                       prepare_data_behavior_ins.category_count,
                                       self.FLAGS.max_len)

        self.train_set, self.test_set = prepare_data_behavior_ins.get_train_test()
        self.logger.info('Get Train Test Data Process.\tCost time: %.2fs' % (time.time() - start_time))

        # self.item_category_dic = prepare_data_behavior_ins.item_category_dic
        self.global_step = 0
        self.one_epoch_step = 0
        self.now_epoch = 0

    '''
    def _eval_auc(self, test_set):
      auc_input = []
      auc_input = np.reshape(auc_input,(-1,2))
      for _, uij in DataInputTest(test_set, FLAGS.test_batch_size):
        #auc_sum += model.eval(sess, uij) * len(uij[0])
        auc_input = np.concatenate((auc_input,self.model.eval_test(self.sess,uij)))
      #test_auc = auc_sum / len(test_set)
      test_auc = roc_auc_score(auc_input[:,1],auc_input[:,0])

      self.model.eval_writer.add_summary(
          summary=tf.Summary(
              value=[tf.Summary.Value(tag='New Eval AUC', simple_value=test_auc)]),
          global_step=self.model.global_step.eval())

      return test_auc
      '''

    def train(self):

        start_time = time.time()
        # Config GPU options
        if self.FLAGS.per_process_gpu_memory_fraction == 0.0:
            gpu_options = tf.GPUOptions(allow_growth=True)
        elif self.FLAGS.per_process_gpu_memory_fraction == 1.0:
            gpu_options = tf.GPUOptions()

        else:
            gpu_options = tf.GPUOptions(
                per_process_gpu_memory_fraction=self.FLAGS.per_process_gpu_memory_fraction)

        os.environ['CUDA_VISIBLE_DEVICES'] = self.FLAGS.cuda_visible_devices

        # Initiate TF session
        # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        global_step_lr = tf.Variable(0, trainable=False)
        learing_rate = tf.train.exponential_decay(
            learning_rate=self.FLAGS.learning_rate, global_step=global_step_lr, decay_steps=100, decay_rate=0.995,
            staircase=True)
        with self.sess.as_default():

            # Create a new model or reload existing checkpoint
            if self.FLAGS.experiment_type == "atrank":
                self.model = Atrank_model(self.FLAGS, self.emb, self.sess)

            elif self.FLAGS.experiment_type == "slirec":
                self.model = SLiRec_model(self.FLAGS, self.emb, self.sess)

            elif self.FLAGS.experiment_type == "bpr":
                self.model = Bprmf_model(self.FLAGS, self.emb,self.sess)

            elif self.FLAGS.experiment_type == 'dib':
                self.model = DIB_model(self.FLAGS, self.emb,self.sess)

            elif self.FLAGS.experiment_type == 'no_emb_dib':
                self.model = DIB_model(self.FLAGS, self.emb,self.sess)

            elif self.FLAGS.experiment_type == 'sasrec':
                self.model = SASRec_model(self.FLAGS, self.emb,self.sess)

            elif self.FLAGS.experiment_type == 'grurec':
                self.model = GRU4Rec_model(self.FLAGS, self.emb,self.sess)

            elif self.FLAGS.experiment_type == 'lstur':
                self.model = LSTUR_model(self.FLAGS, self.emb,self.sess)

            elif self.FLAGS.experiment_type == 'bert':
                self.model = BERT4Rec_model(self.FLAGS, self.emb,self.sess)

            elif self.FLAGS.experiment_type == 'dmpn':
                self.model = DMPN_model(self.FLAGS, self.emb,self.sess)


            self.logger.info('Init model finish.\tCost time: %.2fs' % (time.time() - start_time))

            # test_auc = self.model.metrics(sess=self.sess,
            #                               batch_data=self.test_set,
            #                               global_step=self.global_step,
            #                               name='test auc')

            # Eval init AUC
            # self.logger.info('Init AUC: %.4f' % test_auc)

            recall_rate, avg_ndcg = self.model.metrics_topK(sess=self.sess,
                                                            batch_data=self.test_set,
                                                            global_step=self.global_step,
                                                            topk=self.FLAGS.top_k)

            self.logger.info('Init recall_rate: %.4f' % recall_rate)
            self.logger.info('Init avg_ndcg: %.4f' % avg_ndcg)

            # Start training
            self.logger.info('Training....\tmax_epochs:%d\tepoch_size:%d' % (self.FLAGS.max_epochs,self.FLAGS.train_batch_size))
            start_time = time.time()
            avg_loss = 0.0
            self.best_hr_5, self.best_ndcg_5,\
            self.best_hr_10, self.best_ndcg_10,\
            self.best_hr_20, self.best_ndcg_20, = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            for epoch in range(self.FLAGS.max_epochs):

                random.shuffle(self.train_set)
                self.logger.info('train_set:%d\t test_set:%d' % (len(self.train_set), len(self.test_set)))

                epoch_start_time = time.time()

                for step_i, train_batch_data in DataInput(self.train_set, self.FLAGS.train_batch_size):
                    try:
                        lr = self.sess.run(learing_rate, feed_dict={global_step_lr: self.global_step})
                        add_summary = bool(self.global_step % self.FLAGS.display_freq == 0)
                        step_loss, merge = self.model.train(self.sess,train_batch_data,lr,add_summary,self.global_step)
                        self.model.train_writer.add_summary(merge,self.global_step)
                        avg_loss = avg_loss + step_loss
                        self.global_step = self.global_step + 1
                        self.one_epoch_step = self.one_epoch_step + 1

                        #evaluate for eval steps
                        if self.global_step % self.FLAGS.eval_freq == 0:

                            self.logger.info("epoch_step:%d  global_step:%d batch_loss:%.4f" % (self.one_epoch_step,
                                                                                                self.global_step,
                                                                                                (avg_loss / self.FLAGS.eval_freq)))

                            # train_auc = self.model.metrics(sess=self.sess, batch_data=train_batch_data,
                            #                               global_step=self.global_step,name='train auc')
                            # self.logger.info('Batch Train AUC: %.4f' % train_auc)
                            # self.test_auc = self.model.metrics(sess=self.sess, batch_data=self.test_set,
                            #                               global_step=self.global_step,name='test auc')
                            # self.logger.info('Test AUC: %.4f' % self.test_auc)
                            # self.recall_rate, self.avg_ndcg = self.model.metrics_topK(sess=self.sess, batch_data=self.test_set,
                            #                                                 global_step=self.global_step, topk=self.FLAGS.top_k)
                            self.hr_5, self.ndcg_5, \
                            self.hr_10, self.ndcg_10, \
                            self.hr_20, self.ndcg_20 = self.model.test(self.sess, self.test_set, self.global_step)

                            self.logger.info('HR@5: %.4f NDCG@5: %.4f\t'
                                             'HR@10: %.4f NDCG@10: %.4f\t'
                                             'HR@20: %.4f NDCG@20: %.4f' % (self.hr_5, self.ndcg_5,
                                                                            self.hr_10, self.ndcg_10,
                                                                            self.hr_20, self.ndcg_20))
                            avg_loss = 0

                            self.save_model()
                            if self.FLAGS.draw_pic == True:
                                self.save_fig()

                    except Exception as e:
                        self.logger.info("Error！！！！！！！！！！！！")
                        self.logger.info(e)


                self.logger.info('one epoch Cost time: %.2f' %(time.time() - epoch_start_time))

                #evaluate test auc and train auc for an epoch
                # test_auc = self.model.metrics(sess=self.sess, batch_data=self.test_set,
                #                                       global_step=self.global_step,name='test auc')
                # self.logger.info('Test AUC for epoch %d: %.4f' % (epoch, test_auc))

                self.one_epoch_step = 0
                # if self.global_step > 1000:
                #     lr = lr / 2
                # elif lr < 10e-5:
                #     lr = lr * 0.88
                # else:
                #     lr = lr * 0.95

                self.logger.info('Epoch %d DONE\tCost time: %.2f' % (self.now_epoch, time.time() - start_time))
                self.logger.info("----------------------------------------------------------------------")

                self.now_epoch = self.now_epoch + 1
                self.one_epoch_step = 0


        self.model.save(self.sess,self.global_step)
        # self.logger.info('best test_auc: ' + str(self.best_auc))
        self.logger.info('best HR@5: ' + str(self.best_hr_5))
        self.logger.info('best HR@10: ' + str(self.best_hr_10))
        self.logger.info('best HR@20: ' + str(self.best_hr_20))

        self.logger.info('best NDCG@5: ' + str(self.best_ndcg_5))
        self.logger.info('best NDCG@10: ' + str(self.best_ndcg_10))
        self.logger.info('best NDCG@20: ' + str(self.best_ndcg_20))

        self.logger.info('Finished')

    #judge to save model
    #three result for evaluating model: auc ndcg recall
    def save_model(self):

        #  avg_loss / self.FLAGS.eval_freq, test_auc,test_auc_new))
        # result.append((self.model.global_epoch_step.eval(), model.global_step.eval(), avg_loss / FLAGS.eval_freq, _eval(sess, test_set, model), _eval_auc(sess, test_set, model)))avg_loss = 0.0
        # only store good model

        is_save_model = False
        #for bsbe
        # if self.FLAGS.experiment_type == "bsbe" or self.FLAGS.experiment_type == "bpr":
        # if self.FLAGS.experiment_type == "bsbe":
        #     if (self.test_auc > 0.85 and self.test_auc - self.best_auc > 0.01):
        #         self.best_auc = self.test_auc
        #         is_save_model = True
        #
        # #recall  for  istsbp
        # elif self.FLAGS.experiment_type == "istsbp" or self.FLAGS.experiment_type == "pistrec":
        #     if self.recall_rate > 0.15 and self.recall_rate > self.best_recall:
        #         self.best_recall = self.recall_rate
        #         is_save_model =True

        if self.hr_5 > self.best_hr_5:
            self.best_hr_5 = self.hr_5
        if self.hr_10 > self.best_hr_10:
            self.best_hr_10 = self.hr_10
        if self.hr_20 > self.best_hr_20:
            self.best_hr_20 = self.hr_20

        if self.ndcg_5 > self.best_ndcg_5:
            self.best_ndcg_5 = self.ndcg_5
        if self.ndcg_10 > self.best_ndcg_10:
            self.best_ndcg_10 = self.ndcg_10
        if self.ndcg_20 > self.best_ndcg_20:
            self.best_ndcg_20 = self.ndcg_20

        if self.global_step % 1000 == 0:
            is_save_model = True

        if is_save_model == True:
            self.model.save(self.sess, self.global_step)



if __name__ == '__main__':
    main_process = Train_main_process()
    main_process.train()
