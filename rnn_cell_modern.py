"""Module for constructing RNN Cells"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math, numpy as np
from six.moves import xrange 
import tensorflow as tf

# from multiplicative_integration import multiplicative_integration, multiplicative_integration_for_multiple_inputs

from tensorflow.python.ops.nn import rnn_cell
import highway_network
RNNCell = rnn_cell.RNNCell


class HighwayRNNCell(RNNCell):
  """Highway RNN Network with multiplicative_integration"""

  def __init__(self, num_units, num_highway_layers = 3):
    self._num_units = num_units
    self.num_highway_layers = num_highway_layers


  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, timestep = 0, scope=None):
    current_state = state
    for highway_layer in xrange(self.num_highway_layers):
      with tf.variable_scope('highway_factor_'+str(highway_layer)):
        highway_factor = tf.tanh(tf.rnn_cell._linear([inputs, current_state], self._num_units, True))
      with tf.variable_scope('gate_for_highway_factor_'+str(highway_layer)):
        gate_for_highway_factor = tf.sigmoid(tf.rnn_cell._linear([inputs, current_state], self._num_units, True, -3.0))

        gate_for_hidden_factor_= 1 - gated_for_highway_factor

      current_state = highway_factor * gated_for_highway_factor + current_state * gated_for_hidden_factor

    return current_state, current_state


