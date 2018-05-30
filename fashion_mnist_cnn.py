# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:05:23 2018

@author: cbel-amira
"""

import tensorflow as tf
import numpy as np
import csv
import operator

layers = tf.contrib.layers

mode = False#false: training, true: test
resume_training = False

input_data_path = './data/'
save_model_path = './model/save_net.ckpt'
output_csv_file = 'result_new.csv'

batch_size = 50
display_step = 100
training_iters = 40000


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(input_data_path, one_hot=True)

tf.reset_default_graph()

x = tf.placeholder(tf.float32,[None, 784])#784 = 28*28
y = tf.placeholder(tf.float32,[None, 10])

x_img = tf.reshape(x, [-1,28,28,1])

output = layers.conv2d(x_img, 32, [5,5], stride=1)
output = tf.nn.relu(output)
#output = layers.conv2d(output, 64, [5,5], stride=1)
#output = tf.nn.relu(output)
output = tf.nn.max_pool(output,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


output = layers.conv2d(x_img, 64, [5,5], stride=1)
output = tf.nn.relu(output)
#output = layers.conv2d(output, 128, [5,5], stride=1)
#output = tf.nn.relu(output)
output = tf.nn.max_pool(output,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

output = layers.flatten(output)
output = layers.fully_connected(output,1024,normalizer_fn=None,activation_fn=tf.nn.relu)

keep_prob = tf.placeholder(tf.float32)
output = tf.nn.dropout(output,keep_prob)

output = layers.flatten(output)
output = layers.fully_connected(output,10,normalizer_fn=None,activation_fn=None)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=output))
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_pred = tf.equal(tf.arg_max(output,1),tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32)) 

save = tf.train.Saver()

if mode == False:
    with tf.Session() as sess:
        if resume_training == False:
            sess.run(tf.global_variables_initializer())
        else:
            save.restore(sess, save_model_path)
            print("resume training model")
            
        for i in range(training_iters):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            if i % display_step == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
                print("step = %d, training_accuracy = %g" %(i,train_accuracy))
                sess.run(train_op,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.5})
                
                
        print("test accuracy = %g" % accuracy.eval(feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0}))
        save_path = save.save(sess, save_model_path)
        print("save model",save_path)
else:
    with tf.Session() as sess:
        save.restore(sess, save_model_path)
        print("start testing model") 

        predict_te = sess.run(output,feed_dict={x:mnist.test.images,keep_prob:1.0})
        
        #get the max of every vector
        pre_teno = []
        
        for k in range(0,len(predict_te),1):
            index, value = max(enumerate(predict_te[k]), key=operator.itemgetter(1))
            pre_teno.append(index)
        
        result = []
        for j in range(0,len(pre_teno),1):
            temp = str(pre_teno[j])
            result.append(temp)
        
        #Save the result into a csv file which will be summitted to kaggle       
        with open(output_csv_file, mode='w',newline='', encoding='utf-8') as write_file:
            writer = csv.writer(write_file)
            writer.writerow(['label'])
            for i in range (0,len(result),1):
                writer.writerow(result[i])

        print("csv_result generated")









