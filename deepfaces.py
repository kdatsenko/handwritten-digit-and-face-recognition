################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
#from pylab import *
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.io import loadmat, savemat
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import tensorflow as tf
from partition_data import fetch_sets

from actor_classes import class_names

N_LABELS = 6 #6 actors
ALEX_IMG_DIM = 227 #used to be 64
TF_IMG_DIM = 13
CONV_DIM = 384
IMG_SIZE = 13*13*384

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 6))
x_dim = (1, 227,227,3)
xdim = train_x.shape[1:]
ydim = train_y.shape[1] 

################################################################################
#Read Image, and change to BGR

def transf_imgs(M, actor):
    M_actor = zeros((0,)+M[actor][0].shape)
    for image in range(len(array(M[actor]))):
        #img = 227*227*3
        img = ((array(M[actor])[image])/255.).astype(float32) 
        img_placeholder = zeros(x_dim).astype(float32)
        img_t = img_placeholder.copy() #img transformed, BGR, mean
        img_t[0,:,:,0], img_t[0,:,:,2] = img[:,:,2], img[:,:,0]
        img_t = img_t-mean(img_t)
        M_actor = vstack((M_actor, img_t))
    return M_actor
    
def get_dataset_alex(M, setname):
    dummy_name = setname + "0"
    
    batch_xs = zeros((0,)+M[dummy_name][0].shape)
    batch_y_s = zeros( (0, N_LABELS))
    
    keys =  [setname+str(i) for i in range(N_LABELS)]
    for k in range(N_LABELS): #each train set k from M
        actor = keys[k]
        M_actor = transf_imgs(M, actor)
        batch_xs = vstack((batch_xs, M_actor))
        one_hot = zeros(N_LABELS)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s, tile(one_hot, (len(M[actor]), 1))))
    return batch_xs, batch_y_s
    
def get_dataset_tf(M, setname):
    dummy_name = setname + "0"
    batch_xs = zeros((0,)+M[dummy_name][0].shape)
    batch_y_s = zeros( (0, N_LABELS))
    keys =  [setname+str(i) for i in range(N_LABELS)]
    for k in range(N_LABELS):
        actor = keys[k]
        #Don't normalize with 255
        batch_xs = vstack((batch_xs, (array(M[actor])[:])  ))
        #batch_xs = vstack((batch_xs, ((array(M[actor])[:])/255.)  ))
        one_hot = zeros(N_LABELS)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[actor]), 1))   ))
    return batch_xs, batch_y_s
   


def get_train_batch(M, N):
    n = int(N/N_LABELS)
    # should be ALEX_IMG_DIM, ALEX_IMG_DIM, 3
    batch_xs = zeros((0,)+M['train0'][0].shape) 
    batch_y_s = zeros( (0, N_LABELS))
    
    train_k =  ["train"+str(i) for i in range(N_LABELS)]
    for k in range(N_LABELS): #for each actor
        train_size = len(M[train_k[k]]) #imgs per actor
        #idx: array of n random indexes between 0 and train_siz
        idx = array(random.permutation(train_size)[:n])
        
        #batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[idx])/255.)  ))
        batch_xs = vstack((batch_xs, (array(M[train_k[k]])[idx])  ))
        one_hot = zeros(N_LABELS)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   ))
    return batch_xs, batch_y_s


################################################################################

#DON'T MODIFY
def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel) 
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
    return  tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])

def get_alexnet_output():
    
    #OLD
    #x = tf.placeholder(tf.float32, (None,) + xdim)
    
    M = loadmat("faces_10.mat")
    #In Python 3.5, change this to:
    net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
    #net_data = load("bvlc_alexnet.npy").item()

    x = tf.placeholder(tf.float32, [1, ALEX_IMG_DIM, ALEX_IMG_DIM, 3])
    
    
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in)
    
    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = tf.nn.local_response_normalization(conv1,
                                                    depth_radius=radius,
                                                    alpha=alpha,
                                                    beta=beta,
                                                    bias=bias)
    
    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    
    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in)
    
    
    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = tf.nn.local_response_normalization(conv2,
                                                    depth_radius=radius,
                                                    alpha=alpha,
                                                    beta=beta,
                                                    bias=bias)
    
    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    
    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in)
    
    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
   
    ################################################################################
    
    #Make new Training, Validation, Testing sets based on conv4 activation
    
    set_names = ['train','valid','test']
    train_x, _ = get_dataset_alex(M, set_names[0])
    valid_x, _ = get_dataset_alex(M, set_names[1])
    test_x, _ = get_dataset_alex(M, set_names[2])
    
    single_conv4_dim = TF_IMG_DIM * TF_IMG_DIM * CONV_DIM

    conv_sets = [np.empty((1, single_conv4_dim)), 
                 np.empty((1, single_conv4_dim)),
                 np.empty((1, single_conv4_dim))]
    
    
    datasets = [train_x, valid_x, test_x]
    for i in range(3):
        set_size = datasets[i].shape[0]
        data_set = datasets[i]
        for j in range(set_size):
            im = sess.run(conv4, feed_dict={x:array([data_set[j]])})
            im = np.reshape(im, (1,single_conv4_dim))
            conv_sets[i]=np.concatenate((conv_sets[i],im))
            
    conv_sets[0] = conv_sets[0][1:]
    conv_sets[1] = conv_sets[1][1:]
    conv_sets[2] = conv_sets[2][1:]
    
    new_M = {}
    for i in range(3):
        set_size = int(conv_sets[i].shape[0]/6)
        for j in range(6):
            idx = set_size * j
            new_M[set_names[i]+str(j)] = conv_sets[i][idx:(idx+set_size)]   
        
    savemat('alex.mat', new_M)
    print('Retrieved Activations from AlexNet')
    
    
def part10():
    if not os.path.exists('alex.mat'):
        get_alexnet_output()
    M = loadmat("alex.mat")
    
    #Get train, test, valid samples
    test_x, test_y = get_dataset_tf(M, 'test')
    valid_x, valid_y = get_dataset_tf(M, 'valid')
    train_x, train_y = get_dataset_tf(M, 'train')  
    
    #One fully connected layer 
    
    x = tf.placeholder(tf.float32, [None, IMG_SIZE])    
    
    W = tf.Variable(tf.random_normal([IMG_SIZE, N_LABELS], stddev=0.01))/10
    b = tf.Variable(tf.random_normal([N_LABELS], stddev=0.01))/10
    
    layer = tf.matmul(x, W)+b
    
    y = tf.nn.softmax(layer)
    y_ = tf.placeholder(tf.float32, [None, N_LABELS])

    lam = 0.0085
    decay_penalty =lam*tf.reduce_sum(tf.square(W))
    NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty
    
    alpha = 0.005
    train_step = tf.train.GradientDescentOptimizer(alpha).minimize(NLL)
    
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    for i in range(3000):
        batch_xs, batch_ys = get_train_batch(M, 50)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        
        if i % 100 == 0:
            val_accuracy = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
            train_accuracy = sess.run(accuracy, feed_dict={x: train_x, y_: train_y})
            test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
            print("i=",i)
            print("Train:", train_accuracy)
            print("Validation:", val_accuracy)
            print("Test:", test_accuracy)
            print("Penalty:", sess.run(decay_penalty))
            
    

    val_accuracy = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
    train_accuracy = sess.run(accuracy, feed_dict={x: train_x, y_: train_y})
    test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
    print("The final performance on the training set is: ", train_accuracy)
    print("The final validation set accuracy is: ", val_accuracy)
    print("The final performance on the test set is: ", test_accuracy)
    

if __name__ == "__main__":
    random.seed(0)
    print("================== RUNNING PART 10 ===================")
    fetch_sets(75, 30, 15)
    part10()
    
    #Part 7: The final performance on the test set is:  0.9
    #In part 10, get it up to at least 0.93, for 30% error rate improvement




