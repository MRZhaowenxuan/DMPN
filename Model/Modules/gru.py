from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
import tensorflow as tf
from Model.Modules.time_aware_gru import T_GRUCell


class GRU():

    def build_single_cell(self, hidden_units):
        ''' Create a single-layer RNNCell.

            Args:
              hidden_units: Units of RNNCell.

            Returns:
             An example of the single layer RNNCell
            '''
        cell_type = GRUCell
        cell = cell_type(hidden_units)
        return cell

    def build_cell(self, hidden_units,
                   depth=1):
        '''Create forward and reverse RNNCell networks.

            Args:
              hidden_units: Units of RNNCell.
              depth: The number of RNNCell layers.

            Returns:
              An example of RNNCell
            '''
        cell_lists = [self.build_single_cell(hidden_units) for i in range(depth)]
        return MultiRNNCell(cell_lists)

    def bidirectional_gru_net(self, hidden_units,
                              input_data,
                              input_length):
        '''Create GRU net.

            Args:
              hidden_units: Units of RNN.
              input_data: Input of RNN.
              input_length: The length of input_data.

            Returns:
              Outputs, which are (output_fw, output_bw), are composed of tensor
              which outputs outward to cell and outward to cell
            '''
        cell_fw = self.build_cell(hidden_units)
        cell_bw = self.build_cell(hidden_units)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input_data, input_length, dtype=tf.float32)
        output = tf.layers.dense(tf.concat([outputs[0], outputs[1]], -1), 128)

        return output

    def gru_net(self,hidden_units,input_data,input_length):


        cell = self.build_cell(hidden_units)
        input_length = tf.reshape(input_length,[-1])
        outputs, _ = tf.nn.dynamic_rnn(cell,inputs=input_data,sequence_length=input_length,dtype=tf.float32)

        return outputs

    def gru_net_initial(self, hidden_units, input_data, initial_state, input_length, depth=1):
        cell_lists = [self.build_single_cell(hidden_units) for i in range(depth)]
        multi_cell = MultiRNNCell(cell_lists, state_is_tuple=False)
        input_length = tf.reshape(input_length, [-1])
        output, state = tf.nn.dynamic_rnn(multi_cell, input_data, sequence_length=input_length,
                                          initial_state=initial_state, dtype=tf.float32)
        return output

    def build_time_gru_cell(self, hidden_units, depth=1):
        cell = T_GRUCell(hidden_units)
        # cell_lists = [cell for i in range(depth)]
        return cell

    def time_gru_net(self, hidden_units, input_data, initial_state, input_length):
        cell = self.build_time_gru_cell(hidden_units)
        self.input_length = tf.reshape(input_length, [-1])
        output, state = tf.nn.dynamic_rnn(cell, input_data, sequence_length=input_length,
                                          initial_state=initial_state, dtype=tf.float32)
        return output



