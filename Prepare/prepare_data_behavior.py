from Prepare.prepare_data_base import prepare_data_base
import numpy as np
from Prepare.mask_data_process import mask_data_process

np.random.seed(1234)


class prepare_data_behavior(prepare_data_base):

    def __init__(self, FLAGS, origin_data):

        #init prepare
        super(prepare_data_behavior, self).__init__(FLAGS, origin_data)
        self.max_len = FLAGS.max_len

    def data_handle_process(self, x):
        #不同的数据处理方式
        behavior_seq = self.data_handle_process_base(x)
        if behavior_seq is None:
            return

        mask_data_process_ins = mask_data_process(behavior_seq=behavior_seq,
                                                  use_action=self.use_action,
                                                  mask_rate=self.mask_rate,
                                                  offset=self.offset)

        time_offset = 0 + self.offset
        action_offset = 2 + self.offset
        mask_data_process_ins.get_mask_index_list_behaivor()

        for index in mask_data_process_ins.mask_index_list:

            #index: the index of behavior_sequence
            # neg_item_list = mask_data_process_ins.get_neg_item(self.item_count, self.neg_sample_ratio)

            # if index < 10:
            #     continue
            if self.max_len <= 10:
                if index < 2:
                    continue
            else:
                if index < 10:
                    continue

            user_id, item_seq_temp, factor_list, time_list = \
                mask_data_process_ins.mask_process_unidirectional(self.FLAGS.causality, index)

            def position(time_list):
                pos_list = [i for i in range(len(time_list))]
                pos_list = np.array(pos_list)
                time_list = np.array(time_list)
                return list(pos_list + time_list)

            #按照时间戳取出时间差
            time_list = [int(x / 3600) for x in time_list]
            # time_list = position(time_list)
            target_time = int(mask_data_process_ins.time_stamp_seq[index] / 3600)
            time_interval_list = mask_data_process_ins.proc_time(position(time_list), target_time)
            def proc_pos(time_list):
                pos_last_list = [len(time_list) - i for i in range(1, len(time_list) + 1)]
                return pos_last_list

            pos_last_list = proc_pos(time_list)

            #update time
            # time_interval_seq = mask_data_process_ins.proc_time_emb(factor_list[1],
            #                                                         mask_data_process_ins.time_stamp_seq[index],
            #                                                         self.gap)
            factor_list[-1] = time_interval_list

            #give end index
            temp_index = len(item_seq_temp) - 1
            # neg_list = []
            if self.use_action == True:

                # time_intervel = 0  action =2
                pos_list = (mask_data_process_ins.item_seq[index],
                            self.item_category_dic[mask_data_process_ins.item_seq[index]], action_offset, time_offset)

                # for neg_item_id in neg_item_list:
                #     neg_list.append((neg_item_id, self.item_category_dic[neg_item_id], time_offset, action_offset))

            else:

                pos_list = (mask_data_process_ins.item_seq[index],
                            self.item_category_dic[mask_data_process_ins.item_seq[index]], time_offset)

                # for neg_item_id in neg_item_list:
                #     neg_list.append((neg_item_id, self.item_category_dic[neg_item_id], time_offset))

            self.data_set.append((user_id, item_seq_temp, factor_list, temp_index, pos_list, time_list, pos_last_list))


    def format_train_test(self):

        self.train_set = []
        self.test_set = []
        for i in range(len(self.data_set)):
            if i % self.test_frac == 0:
                self.test_set.append(self.data_set[i])
            else:
                self.train_set.append(self.data_set[i])






    """
      mask the data as the predict item
     :param
         mask_rate: the rate to mask behaviors (If there is actions, only purchase will be masked.)
         data_size: number of samples

         #only_last: mask only last item
     :returns

    """






