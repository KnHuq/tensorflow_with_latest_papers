from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

class partial_ordering_embedding(object):

	def __init__(self, embedding_size):
		self.embedding_size = embedding_size

