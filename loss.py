# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 11:03:09 2017

@author: 49603
"""


import tensorflow as tf
import numpy as np

from gan_model import *
from sub_function import *

'''placeholder'''
x_real  = tf.placeholder(tf.float32,[None,height,width,c_dim], name='real_images')
x_real_s= tf.placeholder(tf.float32,[None,spec_dim], name='real_images_s')

y_real= tf.placeholder(tf.float32,[None,class_num],name='real_label')
y_fake= tf.placeholder(tf.float32,[None,class_num],name='fake_label')

z  =tf.placeholder(tf.float32,[batch_size,z_dim],name='noise')
z_s=tf.placeholder(tf.float32,[batch_size,z_dim],name='noise_s')

with tf.variable_scope("generator"):
    G = generator(z)
    
with tf.variable_scope("generator_spec"):
    x_real_sr = tf.reshape(x_real_s,[-1,spec_dim,1,1], name='real_images_sr')
    G_s = generator_spec(z_s)

with tf.variable_scope("discriminator") as scope:
    y_real_logit=discriminator(x_real,x_real_sr)
    scope.reuse_variables()
    y_fake_logit=discriminator(G,G_s)

d_loss_real = tf.reduce_mean(cross_entropy( y_real_logit, y_real))
d_loss_fake = tf.reduce_mean(cross_entropy( y_fake_logit, y_fake))

g_loss = tf.reduce_mean(cross_entropy(y_fake_logit, y_real))
d_loss =d_loss_real + d_loss_fake

   
    
    



        

            
