# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:38:07 2017

@author: 49603
"""
import tensorflow as tf 
import numpy as np

from sub_function import *

s_h, s_w=height, width
s_h2, s_w2  = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
s_h4, s_w4  = conv_out_size_same(s_h2,2), conv_out_size_same(s_w2,2)
s_h8, s_w8  = conv_out_size_same(s_h4,2), conv_out_size_same(s_w4,2)
s_h16,s_w16 = conv_out_size_same(s_h8,2), conv_out_size_same(s_w8,2)

s_h_s=spec_dim
s_h2_s = conv_out_size_same(s_h_s, 3)
s_h4_s = conv_out_size_same(s_h2_s,2)
s_h8_s = conv_out_size_same(s_h4_s,2)
s_h16_s= conv_out_size_same(s_h8_s,2)

def generator(z):
    #scope.reuse_variables()
    g_bn0 = batch_norm(name='g_bn0')
    g_bn1 = batch_norm(name='g_bn1')
    g_bn2 = batch_norm(name='g_bn2')
    g_bn3 = batch_norm(name='g_bn3')
          
    # project `z` and reshape
    z_ = linear(z, gf_dim*4*s_h16*s_w16,'g_h0_lin')
        
    h0 = tf.reshape(z_, [-1, s_h16, s_w16, gf_dim *4])
    h0 = tf.nn.relu(g_bn0(h0))
    
    h1 = deconv2d(h0, [batch_size, s_h8, s_w8, gf_dim*4],gf_dim*4,gf_dim*4, name='g_h1')
    h1 = tf.nn.relu(g_bn1(h1))
        
    h2 = deconv2d(h1, [batch_size, s_h4, s_w4, gf_dim*2],gf_dim*4,gf_dim*2, name='g_h2')
    h2 = tf.nn.relu(g_bn2(h2))
        
    h3 = deconv2d(h2, [batch_size, s_h2, s_w2, gf_dim],gf_dim*2,gf_dim, name='g_h3')
    h3 = tf.nn.relu(g_bn3(h3))
        
    h4 = deconv2d(h3, [batch_size, s_h,  s_w,  c_dim],gf_dim,c_dim, name='g_h4')

    return tf.nn.tanh(h4)

def generator_spec(z):
    
    g_bn0 = batch_norm(name='g_bn0_s')
    g_bn1 = batch_norm(name='g_bn1_s')
    g_bn2 = batch_norm(name='g_bn2_s')
    g_bn3 = batch_norm(name='g_bn3_s')
          
    z_ = linear(z, gf_dim*2*s_h16_s,name='g_h0_lin_s')
        
    h0 = tf.reshape(z_, [-1, s_h16_s, 1, gf_dim * 2])
    h0 = tf.nn.relu(g_bn0(h0))
    
    h1 = deconv2d_spec(h0, [batch_size, s_h8_s, 1, gf_dim*2,],gf_dim*2,gf_dim*2, name='g_h1_s')
    h1 = tf.nn.relu(g_bn1(h1))
        
    h2 = deconv2d_spec(h1, [batch_size, s_h4_s, 1, gf_dim],gf_dim*2,gf_dim, name='g_h2_s')
    h2 = tf.nn.relu(g_bn2(h2))
        
    h3 = deconv2d_spec(h2, [batch_size, s_h2_s, 1, gf_dim],gf_dim,gf_dim, name='g_h3_s')
    h3 = tf.nn.relu(g_bn3(h3))
        
    h4 = deconv2d_spec(h3, [batch_size, s_h_s,  1,  1,],gf_dim,1,d_h=3, name='g_h4_s')

    return tf.nn.tanh(h4)
        
def discriminator(image,image_spec):
    
    d_bn1 = batch_norm(name='d_bn1')
    d_bn2 = batch_norm(name='d_bn2')
    d_bn3 = batch_norm(name='d_bn3')
    d_bn4 = batch_norm(name='d_bn4')
    
    d_bn1_s = batch_norm(name='d_bn1_s')
    d_bn2_s = batch_norm(name='d_bn2_s')
    d_bn3_s = batch_norm(name='d_bn3_s')
    d_bn4_s = batch_norm(name='d_bn4_s')
        
    h0 = tf.nn.relu(d_bn1(conv2d(image, c_dim, df_dim, name='d_h0_conv')))
    h1 = tf.nn.relu(d_bn2(conv2d(h0, df_dim,df_dim*2,  name='d_h1_conv')))
    h2 = tf.nn.relu(d_bn3(conv2d(h1, df_dim*2,df_dim*4,name='d_h2_conv')))
    h3 = tf.nn.relu(d_bn4(conv2d(h2, df_dim*4,df_dim*4,name='d_h3_conv')))
    
    h0_s = tf.nn.relu(d_bn1_s(conv2d_spec(image_spec, 1, df_dim, d_h=3, name='d_h0_conv_s')))
    h1_s = tf.nn.relu(d_bn2_s(conv2d_spec(h0_s, df_dim,df_dim, name='d_h1_conv_s')))
    h2_s = tf.nn.relu(d_bn3_s(conv2d_spec(h1_s, df_dim,df_dim*2, name='d_h2_conv_s')))
    h3_s = tf.nn.relu(d_bn4_s(conv2d_spec(h2_s, df_dim*2,df_dim*2, name='d_h3_conv_s')))
    h4= linear(tf.concat([tf.reshape(h3, [-1, df_dim*4*s_h16*s_w16]),tf.reshape(h3_s, [-1, df_dim*2*s_h16_s])],axis=1), class_num,'d_h3_lin_s')
   
    return h4


       
        

