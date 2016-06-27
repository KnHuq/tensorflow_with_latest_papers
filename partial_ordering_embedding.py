from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

class partial_ordering_embedding(object):
	'''partial_ordering_embedding as defined by Vendrov et al. http://arxiv.org/pdf/1511.06361v6.pdf'''

	def __init__(self, embedding_size):
		self.embedding_size = embedding_size

	#TODO: nick, investigate max margin loss 
	def max_margin_loss(self, alpha = 1.0):
		'''equation 3 in paper

		(u, v) is a seen pair
		where (u', v') is an unseen pair

		Loss = E(f(u), f(v)) + max{0.0, alpha - E(f(u'), f(v'))}
		'''
		positive_loss_part = self.partial_order_error(u,v)
		negative_loss_part = tf.maximum(0.0, alpha - self.partial_order_error(u_prime, v_prime))

		loss = tf.reduce_sum(positive_loss_part + negative_loss_part)

		return loss

	def partial_order_error(self, x, y):
		'''
		calculates the following penalty of an ordered pair (x,y) as defined by:
		E(x,y) = ||max(0, y- x||^2
		notice that the error is always positive, imposing a strong prior to antisymmetry

		x has shape [batch_size, vector_size]
		y has shape [batch_size, vector_size]

		returns partial_order_error of shape [batch_size]
		'''

		def euclidean_norm(tensor):
			'''accepts tensor of shape [batch_size x vector_size]'''
			return tf.reduce_sum(tensor**2, 1)

		#no square op is here because in norm calculation we do NOT take sqrt
		return tf.maximum(0, euclidean_norm(y-x)) 





