from pylab import *
from numpy import *
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
from scipy.stats import bernoulli
import urllib
from numpy import random
from mnist_handout import *
import pickle as cPickle
import scipy.stats

import os
from scipy.io import loadmat

M = loadmat("mnist_all.mat")
train_set = 0
train_t = 0
test_set = 0  
test_t = 0

# Part 1 code
# 10 images of each of the digits in one figure
def plot_number():
    fig = plt.figure(1)
    #plt.subplots_adjust(hspace=0.01) 
    #plt.axes('off')
    plt.axis('off')
    for i in range(0, 10):
        digit_name = "train" + str(i)
        for j in range(0, 10):
            plt.subplot(10,10,10*i+j+1).axis('off')

            number = M[digit_name][j*10].reshape((28,28))
            imgplot = plt.imshow(number, cmap=cm.gray)
    #plt.show()
    savefig('part1_10Imgs')
    
# Part 2 code
def compute_softmax(x, W, b):
    '''Return the softmax matrix of any given 28x28 input image (flattened).
    W is a 784x10 matrix of weights
    x is an 784xN  matrix where N is the number of training cases 
    b is a 10x1 vector  
    y is a 10xN, where N is the number of training cases.
    (10x784)(784xN) + (10x1) = 10xN
    '''
    # calculate outputs
    o = dot(W.T, x)+b
    # calculate the softmax matrix
    return softmax(o)
    
def cost(x, W, b, t):
    '''
    t - target
    y - softmax output
    '''
    y = compute_softmax(x, W, b)
    return -sum(t*log(y))
   

# Part 3(b) code
def gradient_Wb(x, W, b, t):
    '''Return the gradient of the cost function with respect to W and b. 
    W - 784x10 matrix of weights
    x - 784xN  matrix where N is the number of training cases 
    b - 10x1 vector  
    y - 10xN, where N is the number of training cases.
    t - 10xN, representing the target outputs using one-hot codes (each col)
    Returns:
    dCdW - 784x10 matrix
    dCdb - 10x1 vector
    '''
    # gradient of cost function is p-t, where t is a 10xM one-hot encoding
    # vector, and p is the softmax of y

    p = compute_softmax(x, W, b)
    dCdy = p - t

    dCdW = dot(x, dCdy.T)
    dCdb = dot(dCdy, ones((dCdy.shape[1], 1)))
    return dCdW, dCdb 
    
def finite_diff_Wij(x, W, b, i, j, t):
    '''
   Returns dW using finite difference approximate of the gradient of the cost
   with respect to W, at coordinate i
   '''
    h_w = zeros(W.shape)
    h_val = 0.00001
    h_w[i][j] = h_val
    finite_diff_w = (cost(x, W + h_w, b, t) - cost(x, W - h_w, b, t))/ (2*h_val)
    return finite_diff_w
    
def finite_diff_Bj(x, W, b, j, t):
    '''
   Returns dW using finite difference approximate of the gradient of the cost
   with respect to W, at coordinate i
   '''
    h_b = zeros(b.shape)
    h_val = 0.00001
    h_b[j] = h_val
    finite_diff_b = (cost(x, W, b + h_b, t) - cost(x, W, b - h_b, t))/ (2*h_val)
    return finite_diff_b
    

def compare_gradient(num_components=10):
    '''Print out the difference in gradient computation with function of part 3 
    and a finite difference approximation function for the same set of data'''
    # Initiate the data for comparison
    np.random.seed(0)
    W = np.random.rand(784, 10)
    W /= W.size
    b = zeros((10, 1))
    learning_rate = 0.001
    
    x = M["test0"][10].T/255.0
    x = x.reshape((784, 1)) #a single training case
    t = zeros((10, 1))
    t[0] = 1
    
    dW, db = gradient_Wb(x, W, b, t)
    
    #get rand sample of components of W...
    i_sample = np.random.randint(0, W.shape[0], num_components)
    j_sample = np.random.randint(0, W.shape[1], num_components)

    for c in range(num_components):
        i = i_sample[c]
        j = j_sample[c]
        fd_dW = finite_diff_Wij(x, W, b, i, j, t)
        fd_db = finite_diff_Bj(x, W, b, j, t)
        
        print("dC/dWij predicted by finite-diff for (i="+repr(i)+", j="+repr(j)+"): "+repr(fd_dW))
        print("dC/dWij computed precisely by gradient func for (i="+repr(i)+", j="+repr(j)+"): "+repr(dW[i][j]))
        print("dC/dbj predicted by finite-diff for j="+repr(j)+": "+repr(fd_db))
        print("dC/dbj computed precisely by gradient func for j="+repr(j)+": "+repr(db[j]))
        print("--------------------------------------------------------------------")
        
  

# Part 4 code
def part4():
    
    W = np.random.rand(784, 10)
    W /= W.size
    b = zeros((10, 1))
    learning_rate = 0.00001
    m = 50
    #Train the neural network you constructed. 
    W, b, iter_xvals, train_rates, test_rates = grad_descent(train_set, train_t, W, b, learning_rate, iter_limit=2000)
    
    #Plot the learning curves.
    plt.figure(2)
    plt.plot(iter_xvals, train_rates, label="Training Accuracy")
    plt.plot(iter_xvals, test_rates, label="Test Accuracy")
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy (%)')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=2, mode="expand", borderaxespad=0.)
    savefig('part4_iteration_vs_accuracy')
    
    #Display the weights going into each of the output units.
    fig = plt.figure(3)
    
    plt.axis('off')
    for j in range(0, 10):
        plt.subplot(1,10,j+1).axis('off')
        number = W[:, j].reshape((28,28))
        imgplot = plt.imshow(number, cmap=cm.gray)
    plt.show()
    savefig("part4_w_digits")

    
def grad_descent(x, t, init_W, init_b, learning_rate, iter_limit=1000):
    '''Returns the optimized W and b parameters for lowest log loss.
    Arguments:
        x - 784xN  matrix where N is the number of training cases 
        t - 10xN matrix representing the target outputs (one-hot)
        init_W - The initial 784x10 weight matrix.
        b - The 10x1 bias vector.
        learning_rate - The rate initialized for computing gradients
        iter_limit - Maximum number of iterations
    Returns:
        W, b: The optimized Weight matrix and bias vector
        The learning curve points recorded as hitrate vs. iteration number, 
        for plotting the learning curve outside of this function
    '''
    EPS = 1e-10   #EPS = 10**(-10)
    prev_w = init_W-10*EPS
    W = init_W.copy()
    b = init_b.copy()
    iter = 0
    train_rates = []
    test_rates = []
    iter_xvals = []
    while norm(W - prev_w) >  EPS and iter < iter_limit:
        prev_w = W.copy()
        W_grad, b_grad = gradient_Wb(x, W, b, t)
        W -= learning_rate * W_grad
        b -= learning_rate * b_grad
        iter += 1
        if (iter % 100 == 0):
            tr_acc, tst_acc = hitrate(W, b)
            iter_xvals.append(iter)
            train_rates.append(tr_acc)
            test_rates.append(tst_acc)
            print("Iteration: " + str(iter) + " Cost: " + str(cost(x, W, b, t)))
            
    return W, b, iter_xvals, train_rates, test_rates 
    
    
def hitrate(W, b):
    '''Return the classification accuracy rate for any W, b gradient params'''
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0

    train_total = train_set.shape[1]
    test_total = test_set.shape[1]
    
    #(10, 65923)
    train_y_class = argmax(compute_softmax(train_set, W, b), axis=0) 
    test_y_class = argmax(compute_softmax(test_set, W, b), axis=0)
    train_t_class = argmax(train_t, axis=0)
    test_t_class = argmax(test_t, axis=0)
    
    result_train = train_y_class - train_t_class
    result_test = test_y_class - test_t_class
    train_correct = train_total - count_nonzero(result_train)
    test_correct = test_total - count_nonzero(result_test)
    
    return train_correct/train_total, test_correct/test_total
    
    
def get_data():

    #initiate lists for training and test arrays
    train_set = M["train"+str(0)].T/255.0
    test_set = M["test"+str(0)].T/255.0
    one_hot_vector = zeros((10, 1))
    one_hot_vector[0] = 1
    N_train = train_set.shape[1]
    N_test = test_set.shape[1]
    train_t = tile(one_hot_vector, (1, N_train))
    test_t = tile(one_hot_vector, (1, N_test))
    
    for i in range(0,10): #digit
        train_set = np.append(train_set, M["train"+str(i)].T/255.0, axis=1)
        test_set = np.append(test_set, M["test"+str(i)].T/255.0, axis=1)
        N_test = M["test"+str(i)].shape[0]
        N_train = M["train"+str(i)].shape[0]
        one_hot_vector = zeros((10, 1))
        one_hot_vector[i] = 1
        train_t = np.append(train_t, tile(one_hot_vector, (1, N_train)), axis=1)
        test_t = np.append(test_t, tile(one_hot_vector, (1, N_test)), axis=1)
        
    return train_set, test_set, train_t, test_t   
        
        
# Part 5 code
def part5():
    ion()    
    theta = array([0.0, 1.5])
    gen_lin_data_1d(theta, 200, 30, 20.0, 250.0, True)


def plot_line(ax, theta, x_min, x_max, color, label, linewidth=1, ls='-'):
    x_grid_raw = arange(x_min, x_max, 0.01)
    x_grid = vstack((    ones_like(x_grid_raw),
                         x_grid_raw,
                    ))
    y_grid = dot(theta, x_grid)
    ax.plot(x_grid[1,:], y_grid, linestyle=ls, color=color, label=label, linewidth=linewidth)


def gen_lin_data_1d(theta, train_N, test_N, sigma_low, sigma_high, show_plot=True):
    
    random.seed(0)
    
    #####################################################
    # Set up Plots
    
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)

    #####################################################
    # Actual data
    # 2D data points scattered in either halfspace of theta line
    
    x_train_raw = 100*(random.random((train_N))-.5)
    x_test_raw = 100*(random.random((test_N))-.5)
    
    x_train = vstack((    ones_like(x_train_raw),
                    x_train_raw,
                    ))
    x_test = vstack((    ones_like(x_test_raw),
                    x_test_raw,
                    ))
    
    #indices of points with "large outputs" - i.e. far from the theta line
    #10% of points have large outputs
    scatter_train = bernoulli.rvs(0.05, size=train_N) #1 for no flip
    scatter_test = bernoulli.rvs(0.05, size=test_N)

    #95% of y points scattered closer to theta line
    y_train = dot(theta, x_train) + (1.0-scatter_train)*scipy.stats.norm.rvs(scale=sigma_low,size=train_N) + (scatter_train)*scipy.stats.norm.rvs(scale=sigma_high,size=train_N)
    y_test = dot(theta, x_test) + (1.0-scatter_test)*scipy.stats.norm.rvs(scale=sigma_low,size=test_N) + (scatter_test)*scipy.stats.norm.rvs(scale=sigma_high,size=test_N)
    
    #####################################################
    # Label generating process (Two labels: 0 or 1)
    #
    
    #threshold 0, assign label 1 or 0
    t_train = 1  * ((y_train - dot(theta, x_train)) >= 0) 
    t_test = 1  * ((y_test - dot(theta, x_test)) >= 0) 
   
    
    X_train = vstack((x_train, y_train))
    X_test = vstack((x_test, y_test))
    
    if show_plot:
        plot_line(ax1, theta, -70, 70, 'orange', "Actual generating process", linewidth=3, ls=':')
        plot_line(ax2, theta, -70, 70, 'orange', "Actual generating process", linewidth=3, ls=':')
    
        positive_samples_train = X_train[:, where(t_train == 1)]
        negative_samples_train = X_train[:, where(t_train == 0)]
        positive_samples_test = X_test[:, where(t_test == 1)]
        negative_samples_test = X_test[:, where(t_test == 0)]
        
        ax1.plot(negative_samples_train[1,:], negative_samples_train[2,:], "ro")   
        ax1.plot(positive_samples_train[1,:], positive_samples_train[2,:], "go")  
        ax2.plot(negative_samples_test[1,:], negative_samples_test[2,:], "ro")   
        ax2.plot(positive_samples_test[1,:], positive_samples_test[2,:], "go")  

    
    #######################################################
    # Least squares solution
    #
    theta0 = array([0.0, 0.0, 0.0])
    theta_lin = simple_grad_descent(linreg_f, linreg_df, X_train[1:,:], t_train, theta0, 0.00010, 100000)
 
    if show_plot:
        lin_t = array([(0.5 - theta_lin[0])/theta_lin[2], -1*theta_lin[1]/theta_lin[2]])
        plot_line(ax1, lin_t, -70, 70, "k", "Linear Regression Soln")
        plot_line(ax2, lin_t, -70, 70, "k", "Linear Regression Soln")
    
    #######################################################
    # Logistic Regression solution
    # 
    
    theta_log = simple_grad_descent(logreg_f, logreg_df, X_train[1:,:], t_train, theta0, 0.000010, 10000)
    if show_plot:
        log_t = array([(0.5 - theta_log[0])/theta_log[2], -1*theta_log[1]/theta_log[2]])
        plot_line(ax1, log_t, -70, 70, "b", "Logistic Regression Soln")
        plot_line(ax2, log_t, -70, 70, "b", "Logistic Regression Soln")
    #print(logreg_f(X[1:,:], t, theta_log))
    
    #######################################################
    # Classification Results 
    # 
    result_lin = 1 * (dot(theta_lin,X_test) >= 0.5)
    result_log = 1 * (dot(theta_log,X_test) >= 0.5)
    
    classif_lin = asarray([1 if result_lin[i] == t_test[i] else 0 for i in range(len(result_lin))])
    classif_log = asarray([1 if result_log[i] == t_test[i] else 0 for i in range(len(result_lin))])
    
    if show_plot:
        ax1.plot(negative_samples_train[1,0], negative_samples_train[2,0], "ro", label="Negative")   
        ax1.plot(positive_samples_train[1,0], positive_samples_train[2,0], "go", label="Positive") 
        negative_samples_lin = X_test[:, where(classif_lin == 0)]
        negative_samples_log = X_test[:, where(classif_log == 0)]
        ax2.plot(negative_samples_lin[1,:] + 1e-1, negative_samples_lin[2,:] + 1e-1, "ko", mfc='none') 
        ax2.plot(negative_samples_lin[1,0] + 1e-1, negative_samples_lin[2,0] + 1e-1, "ko", mfc='none', label="Lin Reg Miss") 
        ax2.plot(negative_samples_log[1,:] - 1e-1, negative_samples_log[2,:] - 1e-1, "bo", mfc='none') 
        ax2.plot(negative_samples_log[1,0] + 1e-1, negative_samples_log[2,0] + 1e-1, "bo", mfc='none', label="Log Reg Miss") 
        
        ax1.set_title('Training Set')
        ax2.set_title('Test Set')  
        
        # Adds the legend with some customizations.
        legend1 = ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, shadow=True)
        legend2 = ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, shadow=True)
        

        
        frame1 = legend1.get_frame()
        frame1.set_facecolor('0.90')
        frame2 = legend2.get_frame()
        frame2.set_facecolor('0.90')
        
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        savefig('part5_experiment', bbox_inches='tight')
     
    
    lin_hits = sum(classif_lin)
    log_hits = sum(classif_log)
    
    print("\n--------------------------------------------------------")
    print("Results for Part 5 experiment (2 labels for the target 0/1):")
    print("Experiment of training set size " + str(train_N) + " and test set size " + str(test_N) + ".")
    print("Linear Regression correctly classified " + str(lin_hits/float(test_N) * 100) + "% of test set")
    print("Logistic Regression correctly classified " + str(log_hits/float(test_N) * 100) + "% of test set")

def linreg_f(x, y, theta):
    #adds a row of 1s for the bias term on top
    #each column of the X matrix corresponds to a training example 
    x = vstack( (ones((1, x.shape[1])), x)) 
    N = x.shape[1]
    #linear regression cost function
    #the dot (mat-mult) creates a horizontal vector 
    #thus the y vector should be horizontal as well
    return sum( (y - dot(theta.T,x)) ** 2) / N

def linreg_df(x, y, theta):
    #simple gradient of the multivariable linear sum of sqrs cost function
    #applied for multivariable calcs.
    x = vstack( (ones((1, x.shape[1])), x))
    N = x.shape[1]
    return -2*sum((y-dot(theta.T, x))*x, 1) / N #dot(x, (dot(theta, x) - y).T)/N#
    
def logreg_f(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    N = x.shape[1]
    h = 1.0 / (1.0 + exp(-1.0*dot(theta, x)) + 1e-10)
    return -1 * sum( (log(h) * y) + (1.0 - y)*log(1.0 - h) )
    
def logreg_df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    N = x.shape[1]
    h = 1.0 / (1.0 + exp(-1.0*dot(theta, x))) #horizontal
    return -1 * sum((y - h)*x, 1) #dot(x, (h.T - y)) / N
   
def simple_grad_descent(f, df, x, y, init_t, alpha, max_iter):
    EPS = 1e-9   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t) 
        # if iter % 500 == 0:
        #     print (iter)
        #     print (f(x, y, t))
        #     print (df(x, y, t))
        iter += 1
    return t   #return optimal theta  
    

    
if __name__ == "__main__":
    
    random.seed(0)
    train_set, test_set, train_t, test_t = get_data()
    
    #Part 1
    print("================== RUNNING PART 1 ===================")
    plot_number()
    
    #Part 3(b)
    #Check that the gradient was computed correctly by approximating 
    #the gradient at several coordinates using finite differences
    print("================== RUNNING PART 3(b) ===================")
    print("Approximating the gradient at several coordinates using finite differences.")
    compare_gradient()
    print("\n\n")
    
    #Part 4
    print("================== RUNNING PART 4 ===================")
    print("Training the neural network.")
    part4()
    
    #Part 5
    print("================== RUNNING PART 5 ===================")
    part5()
    
    