# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 23:27:38 2017

@author: 49603
"""

import math
import numpy as np 
import tensorflow as tf

def conv_out_size_same(size,  stride):
  return int(math.ceil(float(size) / float(stride)))
  
  
class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)


def conv2d(input_, input_channel, output_channel, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.1,name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_channel,output_channel],initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_channel], initializer=tf.constant_initializer(0.0))
    conv=tf.nn.bias_add(conv, biases)

    return conv

def deconv2d(input_, output_shape,input_channel,output_channel,k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,name=None):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, output_channel, input_channel],initializer=tf.random_normal_initializer(stddev=stddev))
    
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,strides=[1, d_h, d_w, 1])
    
    biases = tf.get_variable('biases', [output_channel], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    return deconv

def conv2d_spec(input_, input_channel, output_channel, k_h=5, k_w=1, d_h=2, d_w=1, stddev=0.1,name="conv2d_spec"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_channel,output_channel],initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_channel], initializer=tf.constant_initializer(0.0))
    conv=tf.nn.bias_add(conv, biases)

    return conv

def deconv2d_spec(input_, output_shape,input_channel,output_channel,k_h=5, k_w=1, d_h=2, d_w=1, stddev=0.02,name=None):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, output_channel, input_channel],initializer=tf.random_normal_initializer(stddev=stddev))
    
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,strides=[1, d_h, d_w, 1])
    
    biases = tf.get_variable('biases', [output_channel], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    return deconv
  
     
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size,name=None,stddev=0.02, bias_start=0.0):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(name):
      matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,tf.random_normal_initializer(stddev=stddev))
      bias = tf.get_variable("bias", [output_size],initializer=tf.constant_initializer(bias_start))
   
      return tf.matmul(input_, matrix) + bias

def cross_entropy(x, y):
     
    return tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)
    

 
  
    
    
