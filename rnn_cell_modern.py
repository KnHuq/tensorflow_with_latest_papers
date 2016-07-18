"""Module for constructing RNN Cells"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math, numpy as np
from six.moves import xrange 
import tensorflow as tf

# from multiplicative_integration import multiplicative_integration, multiplicative_integration_for_multiple_inputs

from tensorflow.python.ops.nn import rnn_cell
import highway_network_modern
from linear_modern import linear

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
        highway_factor = tf.tanh(linear([inputs, current_state], self._num_units, True))
      with tf.variable_scope('gate_for_highway_factor_'+str(highway_layer)):
        gate_for_highway_factor = tf.sigmoid(linear([inputs, current_state], self._num_units, True, -3.0))

        gate_for_hidden_factor = 1 - gate_for_highway_factor

      current_state = highway_factor * gate_for_highway_factor + current_state * gate_for_hidden_factor

    return current_state, current_state


class JZS1Cell(RNNCell):
  """Mutant 1 of the following paper: http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf"""

  def __init__(self, num_units, gpu_for_layer = 0, weight_initializer = "uniform_unit", orthogonal_scale_factor = 1.1):
    self._num_units = num_units
    self._gpu_for_layer = gpu_for_layer 
    self._weight_initializer = weight_initializer
    self._orthogonal_scale_factor = orthogonal_scale_factor

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with tf.device("/gpu:"+str(self._gpu_for_layer)):
      """JZS1, mutant 1 with n units cells."""
      with tf.variable_scope(scope or type(self).__name__):  # "JZS1Cell"
        with tf.variable_scope("Zinput"):  # Reset gate and update gate.
          # We start with bias of 1.0 to not reset and not update.
          '''equation 1 z = sigm(WxzXt+Bz), x_t is inputs'''

          z = tf.sigmoid(linear([inputs], 
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer, orthogonal_scale_factor = self._orthogonal_scale_factor)) 

        with tf.variable_scope("Rinput"):
          '''equation 2 r = sigm(WxrXt+Whrht+Br), h_t is the previous state'''

          r = tf.sigmoid(linear([inputs,state],
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer, orthogonal_scale_factor = self._orthogonal_scale_factor))
          '''equation 3'''
        with tf.variable_scope("Candidate"):
          component_0 = linear([r*state], 
                            self._num_units, True) 
          component_1 = tf.tanh(tf.tanh(inputs) + component_0)
          component_2 = component_1*z
          component_3 = state*(1 - z)

        h_t = component_2 + component_3

      return h_t, h_t #there is only one hidden state output to keep track of. 
      #This makes it more mem efficient than LSTM


class JZS2Cell(RNNCell):
  """Mutant 2 of the following paper: http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf"""

  def __init__(self, num_units, gpu_for_layer = 0, weight_initializer = "uniform_unit", orthogonal_scale_factor = 1.1):
    self._num_units = num_units
    self._gpu_for_layer = gpu_for_layer 
    self._weight_initializer = weight_initializer
    self._orthogonal_scale_factor = orthogonal_scale_factor

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with tf.device("/gpu:"+str(self._gpu_for_layer)):
      """JZS2, mutant 2 with n units cells."""
      with tf.variable_scope(scope or type(self).__name__):  # "JZS1Cell"
        with tf.variable_scope("Zinput"):  # Reset gate and update gate.
          '''equation 1'''

          z = tf.sigmoid(linear([inputs, state], 
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer, orthogonal_scale_factor = self._orthogonal_scale_factor))

          '''equation 2 '''
        with tf.variable_scope("Rinput"):
          r = tf.sigmoid(inputs+(linear([state],
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer, orthogonal_scale_factor = self._orthogonal_scale_factor)))
          '''equation 3'''

        with tf.variable_scope("Candidate"):

          component_0 = linear([state*r,inputs],
                            self._num_units, True)
          
          component_2 = (tf.tanh(component_0))*z
          component_3 = state*(1 - z)

        h_t = component_2 + component_3

      return h_t, h_t #there is only one hidden state output to keep track of. 
        #This makes it more mem efficient than LSTM

class JZS3Cell(RNNCell):
  """Mutant 3 of the following paper: http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf"""

  def __init__(self, num_units, gpu_for_layer = 0, weight_initializer = "uniform_unit", orthogonal_scale_factor = 1.1):
    self._num_units = num_units
    self._gpu_for_layer = gpu_for_layer 
    self._weight_initializer = weight_initializer
    self._orthogonal_scale_factor = orthogonal_scale_factor

  @property
  def input_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with tf.device("/gpu:"+str(self._gpu_for_layer)):
      """JZS3, mutant 2 with n units cells."""
      with tf.variable_scope(scope or type(self).__name__):  # "JZS1Cell"
        with tf.variable_scope("Zinput"):  # Reset gate and update gate.
          # We start with bias of 1.0 to not reset and not update.
          '''equation 1'''

          z = tf.sigmoid(linear([inputs, tf.tanh(state)], 
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer, orthogonal_scale_factor = self._orthogonal_scale_factor))

          '''equation 2'''
        with tf.variable_scope("Rinput"):
          r = tf.sigmoid(linear([inputs, state],
                            self._num_units, True, 1.0, weight_initializer = self._weight_initializer, orthogonal_scale_factor = self._orthogonal_scale_factor))
          '''equation 3'''
        with tf.variable_scope("Candidate"):
          component_0 = linear([state*r,inputs],
                            self._num_units, True)
          
          component_2 = (tf.tanh(component_0))*z
          component_3 = state*(1 - z)

        h_t = component_2 + component_3

      return h_t, h_t #there is only one hidden state output to keep track of. 
      #This makes it more mem efficient than LSTM
