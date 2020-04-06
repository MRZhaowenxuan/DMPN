from Prepare.prepare_data_base import prepare_data_base
import numpy as np
from Prepare.mask_data_process import mask_data_process

np.random.seed(1234)


class prepare_data_pop(prepare_data_base):

    def __init__(self, FLAGS, origin_data):

        #init prepare
        super(prepare_data_pop, self).__init__(FLAGS, origin_data)


    def data_handle_process(self,x):

        behavior_seq = self.data_handle_process_base(x)
        # print(behavior_seq)
        if behavior_seq is None:
            return
        # user_seq = behavior_seq["user_id"].tolist()
        # item_seq = behavior_seq["item_id"].tolist()
        # action_seq = behavior_seq["action_type"].tolist()
        user_id = behavior_seq["user_id"].tolist()[0]
        item_seq = behavior_seq["item_id"].tolist()
        target_item = item_seq[-1]
        item_seq.pop()
        self.data_set.append((user_id, item_seq, target_item))

    def format_train_test(self):

        self.train_set = []
        self.test_set = []
        self.train_set = self.data_set
        self.test_set = self.data_set






    """
      mask the data as the predict item
     :param
         mask_rate: the rate to mask behaviors (If there is actions, only purchase will be masked.)
         data_size: number of samples

         #only_last: mask only last item
     :returns

    """






