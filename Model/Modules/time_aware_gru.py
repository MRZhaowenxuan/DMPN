from keras import activations, initializers
import tensorflow as tf
from keras.activations import sigmoid
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops, init_ops, variable_scope, array_ops, nn_ops
from tensorflow.python.ops.rnn_cell_impl import GRUCell
from tensorflow.python.platform import tf_logging as logging


_BIAS_VARIABLE_NAME = 'bias'
_WEIGHTS_VARIABLE_NAME = 'kernel'


class T_GRUCell(GRUCell):
    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 name=None,
                 dtype=None,
                 **kwargs):
        super(GRUCell, self).__init__(
            _reuse=reuse, name=name, dtype=dtype, **kwargs)

        if context.executing_eagerly() and context.num_gpus() > 0:
            logging.warn("%s: Note that this cell is not optimized for performance. "
                         "Please use tf.contrib.cudnn_rnn.CudnnGRU for better "
                         "performance on GPU.", self)
        # Inputs must be 2-dimensional.
        self.input_spec = input_spec.InputSpec(ndim=2)

        self._num_units = num_units
        if activation:
            self._activation = activations.get(activation)
        else:
            self._activation = math_ops.tanh
        self._kernel_initializer = initializers.get(kernel_initializer)
        self._bias_initializer = initializers.get(bias_initializer)


    def build(self, inputs_shape):
        if inputs_shape[-1] is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % str(inputs_shape))

        input_depth = inputs_shape[-1]
        self._gate_kernel = self.add_variable(
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth-1 + self._num_units, 2 * self._num_units],
            initializer=self._kernel_initializer)
        self._gate_bias = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self._candidate_kernel = self.add_variable(
            "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth-1 + self._num_units, self._num_units],
            initializer=self._kernel_initializer)
        self._candidate_bias = self.add_variable(
            "candidate/%s" % _BIAS_VARIABLE_NAME,
            shape=[self._num_units],
            initializer=(
                self._bias_initializer
                if self._bias_initializer is not None
                else init_ops.zeros_initializer(dtype=self.dtype)))

        self.built = True

    def call(self, inputs, state):
        dtype = inputs.dtype
        time_score = inputs[:, -1]
        time_now_score = tf.expand_dims(inputs[:, -1], -1)
        inputs = inputs[:, :-1]
        input_size = inputs.get_shape().with_rank(2)[1]
        # decay gates
        scope = variable_scope.get_variable_scope()
        with variable_scope.variable_scope(scope) as unit_scope:
            with variable_scope.variable_scope(unit_scope):
                self._time_input_w1 = variable_scope.get_variable(
                    "_time_input_w1", shape=[self._num_units], dtype=dtype, initializer=self._kernel_initializer)
                self._time_input_bias1 = variable_scope.get_variable(
                    "_time_input_bias1", shape=[self._num_units], dtype=dtype,
                    initializer=(
                        self._bias_initializer
                        if self._bias_initializer is not None
                        else init_ops.constant_initializer(1.0, dtype=self.dtype)))
                self._time_kernel_w2 = variable_scope.get_variable(
                    "_time_kernel_w2", shape=[2 * self._num_units, self._num_units], dtype=dtype,
                    initializer=self._kernel_initializer)

        time_now_input = tf.nn.tanh(time_now_score * self._time_input_w1 + self._time_input_bias1)

        time_gate = math_ops.matmul(array_ops.concat([inputs, state], 1), self._time_kernel_w2) + time_now_input
        t = math_ops.sigmoid(tf.square(time_gate))

        inputs_new = t * inputs + (1 - t) * state

        gate_inputs = math_ops.matmul(
            array_ops.concat([inputs_new, state], 1), self._gate_kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._gate_bias)

        value = math_ops.sigmoid(gate_inputs)
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        candidate = math_ops.matmul(
            array_ops.concat([inputs_new, r_state], 1), self._candidate_kernel)
        candidate = nn_ops.bias_add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        #new_h = u * state * sigmoid(time_last_state) + (1 - u) * c * sigmoid(time_now_state)
        new_h = u * state + (1 - u) * c
        return new_h, new_h









