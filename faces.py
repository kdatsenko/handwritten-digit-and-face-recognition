from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
import copy
from scipy.misc import imread
from scipy.misc import imresize
from scipy.io import loadmat, savemat
import matplotlib.image as mpimg
from scipy.ndimage import filters
from partition_data import fetch_sets
import urllib
from numpy import *
import pickle as cPickle
import os
from scipy.io import loadmat
import tensorflow as tf

N_LABELS = 6 #6 actors
IMG_DIM = 64 #repeating Project1
IMG_SIZE = 64*64

act = ['butler', 'radcliffe', 'vartan', 'bracco', 'gilpin', 'harmon'] 


def display_training(M_reg, M_noisy):
    plt.ion()
    fig = plt.figure(1)
    plt.axis('off')
    for j in range(0, N_LABELS):
        plt.subplot(2,N_LABELS,N_LABELS*0+j+1).axis('off')
        img_reg = M_reg["train"+str(j)][0, :].copy()
        reshaped_img_reg = img_reg.reshape((IMG_DIM,IMG_DIM))
        imgplot = plt.imshow(reshaped_img_reg, cmap=cm.gray)
        
        plt.subplot(2,N_LABELS,N_LABELS*1+j+1).axis('off')
        img_noisy = M_noisy["train"+str(j)][0, :].copy()
        reshaped_img_noisy = img_noisy.reshape((IMG_DIM,IMG_DIM))
        imgplot = plt.imshow(reshaped_img_noisy, cmap=cm.gray)
    savefig("part8_noisy_faces")

def apply_noise_to_training(M, display=False):
    train_k =  ["train"+str(i) for i in range(N_LABELS)]
    
    im = M[train_k[0]][0, :]
    
    amount = 0.7
    num_salt = np.ceil(amount * im.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in im.shape]
    
    for k in range(N_LABELS):
        for i in range(M[train_k[k]].shape[0]):
            image = M[train_k[k]][i, :]
            values = [np.random.randint(0, 256) for i in image.shape]
            M[train_k[k]][i, :][coords] = values
    
    if display:
        M1 = loadmat("faces.mat")
        display_training(M1, M)
    return M
    
#M is entire dataset, N is number of images per class label
#Return ???
def get_train_batch(M, N):
    n = int(N/N_LABELS)
    
    batch_xs = zeros((0, M['train0'][0].shape[0]))
    batch_y_s = zeros( (0, N_LABELS))
    
    train_k =  ["train"+str(i) for i in range(N_LABELS)]

    train_size = len(M[train_k[0]])
    
    for k in range(N_LABELS): #for each actor
        train_size = len(M[train_k[k]]) #imgs per actor
        #idx: array of n random indexes between 0 and train_siz
        idx = array(random.permutation(train_size)[:n])
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[idx])/255.)  ))
        one_hot = zeros(N_LABELS)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (n, 1))   ))
    return batch_xs, batch_y_s


def get_test(M):
    batch_xs = zeros((0, M['test0'][0].shape[0]))
    batch_y_s = zeros( (0, N_LABELS))
    
    test_k =  ["test"+str(i) for i in range(N_LABELS)]
    for k in range(N_LABELS):
        batch_xs = vstack((batch_xs, ((array(M[test_k[k]])[:])/255.)  ))
        one_hot = zeros(N_LABELS)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[test_k[k]]), 1))   ))
    return batch_xs, batch_y_s
    
    
def get_train(M):
    batch_xs = zeros((0, M['train0'][0].shape[0]))
    batch_y_s = zeros( (0, N_LABELS))
    
    train_k =  ["train"+str(i) for i in range(N_LABELS)]
    for k in range(N_LABELS):
        batch_xs = vstack((batch_xs, ((array(M[train_k[k]])[:])/255.)  ))
        one_hot = zeros(N_LABELS)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[train_k[k]]), 1))   ))
    return batch_xs, batch_y_s
    
def get_validation(M):
    batch_xs = np.zeros((0, M['valid0'][0].shape[0]))
    batch_y_s = np.zeros( (0, N_LABELS))
    
    valid_k =  ["valid"+str(i) for i in range(N_LABELS)]
    for k in range(N_LABELS):
        batch_xs = vstack((batch_xs, ((array(M[valid_k[k]])[:])/255.)  ))
        one_hot = zeros(N_LABELS)
        one_hot[k] = 1
        batch_y_s = vstack((batch_y_s,   tile(one_hot, (len(M[valid_k[k]]), 1))   ))
    return batch_xs, batch_y_s
    
###############################################################
## Part 7 & 8:

def train_tf_network(M, nhid, lam, num_iter, part7=True):
    #nhid - number of hidden units
    #lam - regularization parameter
    #num_iter - total number of iterations
    #random.seed(0)
    tf.set_random_seed(0)
    
    #nhid = 150 
    
    #Get train, test, valid samples
    test_x, test_y = get_test(M)
    valid_x, valid_y = get_validation(M)
    train_x, train_y = get_train(M)    
    
    x = tf.placeholder(tf.float32, [None, IMG_SIZE])    
    
    W0 = tf.Variable(tf.random_normal([IMG_SIZE, nhid], stddev=0.01))/10
    b0 = tf.Variable(tf.random_normal([nhid], stddev=0.01))/10
    
    W1 = tf.Variable(tf.random_normal([nhid, N_LABELS], stddev=0.01))/10
    b1 = tf.Variable(tf.random_normal([N_LABELS], stddev=0.01))/10
    
    layer1 = tf.nn.tanh(tf.matmul(x, W0)+b0)
    layer2 = tf.matmul(layer1, W1)+b1
    
    y = tf.nn.softmax(layer2)
    y_ = tf.placeholder(tf.float32, [None, N_LABELS])
        
    
    #lam = 0.0
    decay_penalty =lam*tf.reduce_sum(tf.square(W0))+lam*tf.reduce_sum(tf.square(W1))
    NLL = -tf.reduce_sum(y_*tf.log(y))+decay_penalty
    
    #ALPHA
    alpha = 0.05
    train_step = tf.train.GradientDescentOptimizer(alpha).minimize(NLL)
    
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #for plotting
    train_error = []
    val_error = []
    test_error = []
    iter_xvals = []
    
    
    for i in range(num_iter):
 
        batch_xs, batch_ys = get_train_batch(M, 50)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        
        if part7 and i % 100 == 0:
            iter_xvals.append(i)
            val_error.append(accuracy.eval(feed_dict={x: valid_x, y_: valid_y}, session=sess))
            test_error.append(accuracy.eval(feed_dict={x: test_x, y_: test_y}, session=sess))
            train_error.append(accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys}, session=sess))
        if i % 200 == 0: #Print out for user
            print("i=",i)
            # val_accuracy = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
            # train_accuracy = sess.run(accuracy, feed_dict={x: train_x, y_: train_y})
            # test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
            # print("Train:", train_accuracy)
            # print("Validation:", val_accuracy)
            # print("Test:", test_accuracy)
            # print("Penalty:", sess.run(decay_penalty))

    val_accuracy = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
    train_accuracy = sess.run(accuracy, feed_dict={x: train_x, y_: train_y})
    test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})
    print("The final performance on the training set is: ", train_accuracy)
    print("The final validation set accuracy is: ", val_accuracy)
    print("The final performance on the test set is: ", test_accuracy)
    
    if part7:
        plt.figure(2)
        plt.plot(iter_xvals, train_error, label="Training Accuracy")
        plt.plot(iter_xvals, val_error, label="Validation Accuracy")
        plt.plot(iter_xvals, test_error, label="Test Accuracy")
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy (%)')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                ncol=2, mode="expand", borderaxespad=0.)
        savefig('part7_iteration_vs_accuracy')
    
        snapshot = {}
        snapshot["W0"] = sess.run(W0)
        snapshot["W1"] = sess.run(W1)
        snapshot["b0"] = sess.run(b0)
        snapshot["b1"] = sess.run(b1)
        cPickle.dump(snapshot, open("parameters.pkl", "wb"))
 
###############################################################
## Part 9:
    
def visualize_heatmap(W, title, f_num):
    ''' This function is implemented to obtain the heatmaps for the hidden layer
    '''
    fig = figure(f_num)
    ax = fig.gca()
    heatmap = ax.imshow(W.reshape((64, 64)), interpolation='bilinear', cmap= cm.coolwarm)
    fig.colorbar(heatmap, shrink = 0.5, aspect=5)
    fig.suptitle(title, fontsize=20)
    fig.savefig("part9_weight_heatmap_" + title)
    show()
    
def normalize_activations(train, W0, b0, sess):
    """
    Normalize activations for the entire dataset for each neuron.
    """  
    #number of training examples * number of hidden units
    #train: (68, 4096)
    
    X = tf.cast(train, tf.float32)
    activations = tf.nn.tanh(tf.matmul(X, W0)+b0)
    #some neurons fire high for any input
    #normalize them first - normalize all the activations
        
    activ = activations.eval(session=sess)
    mean_activ = np.mean(activ, axis=0)
    std_activ = np.std(activ, axis=0)
    normalized_activ = (activ - mean_activ) / std_activ
    return normalized_activ
    
    
def get_most_sensitive_unit(all_activations, stride, actor):
    from_ind = actor*stride
    to_ind = from_ind + stride
    actor_activ = all_activations[from_ind:to_ind, :]
    
    highest_per_sample = actor_activ.argmax(axis=1)
    most_frequent_max = highest_per_sample
    
    nhid = actor_activ.shape[1]
    frequencies = zeros(nhid)
    for i in range(nhid):
        for unit in highest_per_sample:
            if unit == i:
                frequencies[i] += 1
    return frequencies.argmax()    

def part9():
    # Load network trained from part 1
    snapshot = cPickle.load(open("parameters.pkl", 'rb'))
    
    W0 = tf.Variable(snapshot["W0"])
    b0 = tf.Variable(snapshot["b0"])
    
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    W0_params = W0.eval(session=sess)
    
    M = loadmat("faces.mat")
    
    a1 = 1  #One Male
    a2 = 3  #One Female
    
    actors =  ["train"+str(i) for i in range(N_LABELS)]
    
    #feed forward all actor samples
    #record all the activations in a 300*(training_set_size) array
    
    batch_xs = zeros((0, M['train0'][0].shape[0]))
    for k in range(N_LABELS):
        batch_xs = vstack((batch_xs, ((array(M[actors[k]])[:])/255.)  ))
    
    activ = normalize_activations(batch_xs, W0, b0, sess)
    stride = (array(M[actors[0]])[:]).shape[0]
    
    #Grab most interesting hidden unit
    hidden_unit1 = get_most_sensitive_unit(activ, stride, a1)
    hidden_unit2 = get_most_sensitive_unit(activ, stride, a2)
    
    #Display
    weights_1 = W0_params[:, hidden_unit1]
    weights_2 = W0_params[:, hidden_unit2]
    visualize_heatmap(weights_1, act[a1], 3)
    visualize_heatmap(weights_2, act[a2], 4)
 
 
if __name__ == "__main__":
    
    print("Running Part 8 first, Part 7 second, Part 9 last.\n")
    
    #Part 8
    print("================== RUNNING PART 8 ===================")
    #2000, lam=0.27, alpha = 0.05, nhid=150,
    #prop=0.7 noise
    
    random.seed(0)
    fetch_sets()
    random.seed(0)
    M = loadmat("faces.mat")
    print("Modifying the Training Set to contain 70% noise")
    M_noisy = apply_noise_to_training(M, display=True)
    savemat('faces_noisy.mat', M_noisy)
     
    print("Result of Training for 2000 iterations with NO regularization (lam=0.00)")
    train_tf_network(M_noisy, 150, 0.0, 2000, False)
    print("Result of Training for 2000 iterations WITH regularization (lam=0.23)")
    train_tf_network(M_noisy, 150, 0.24, 2000, False)
    
    random.seed(0)
    #Part 7
    print("================== RUNNING PART 7 ===================")
    fetch_sets(75, 30, 15) #30 images in the Test set
    M = loadmat("faces.mat")
    train_tf_network(M, 300, 0.0, 2000, True)
    print("\n\n")
    
    #Part 9
    print("================== RUNNING PART 9 ===================")
    print("Displaying weights for most sensitive hidden units for 2 actors:")
    part9()
    
    
    
    
    
    
    
    