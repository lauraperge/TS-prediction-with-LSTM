
# coding: utf-8

# # Timeseries forecast with LSTM
# **USDGBP FX rate from December 1979 to March 2018**

# The code is based on:
# 
# https://mapr.com/blog/deep-learning-tensorflow/
# https://github.com/lucko515/tesla-stocks-prediction/blob/master/tensorflow_lstm.ipynb

# ## Dependencies

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import random
get_ipython().run_line_magic('matplotlib', 'inline')
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn
import time
from sklearn.preprocessing import StandardScaler


# ## Data preprocessing

# In[2]:


usdgbp_data_new = pd.read_csv('USD_GBP Historical Data2.csv', header=0)
usdgbp_data_new['Date'] = pd.to_datetime(usdgbp_data_new['Date'])
usdgbp_data_new.head()


# In[3]:


data = usdgbp_data_new['Price']
print('Total number of days in the dataset: {}'.format(len(data)))


# In[4]:


plt.figure(figsize=(12,7), frameon=False, facecolor='brown', edgecolor='blue')
plt.title('Price of USDGBP from end of December 1979 to end of March 2018')
plt.xlabel('Days')
plt.ylabel('FX rate')
plt.plot(data, label='USDGBP FX-rate')
plt.legend()
plt.show()


# ### Train - test split

# * trainset: 8844 (8833 windows), testset: 968; 

# In[5]:


train_end = 8844
train, test = data[:train_end], data[train_end:]


# ### Scaling

# In[6]:


scaler = StandardScaler()
train, test = scaler.fit_transform(train.values.reshape(-1, 1)), scaler.fit_transform(test.values.reshape(-1, 1))


# In[7]:


data_new = np.concatenate((train, test), axis = 0)


# ### Moving Windows

# In[8]:


def window_data(data, window_size):
    X = []
    y = []
    
    i = 0
    while (i + window_size) <= len(data) - 1:
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
        i += 1 
        
        
    assert len(X) ==  len(y)
    return X, y


# In[9]:


window_size = 11
X_train, y_train = window_data(train, window_size)
X_test, y_test = window_data(test, window_size)

#Suitable for neural network
X_train = np.array(X_train).reshape(len(X_train), window_size, 1)
X_test = np.array(X_test).reshape(len(X_test), window_size, 1)

y_train = np.array(y_train).reshape(len(y_train), 1)
y_test = np.array(y_test).reshape(len(y_test), 1)

print("X_train size: {}".format(X_train.shape))
print("y_train size: {}".format(y_train.shape))
print("X_test size: {}".format(X_test.shape))
print("y_test size: {}".format(y_test.shape))


# ## Make RNN 
# - hyperparameters: 
#     * number of hidden layers and their size
#     * batch size
#     * number of epochs
# - regularization: 
#     * dropout wrapper (probabilistic activation on input / recurrent / output)
#     * L1/L2 loss
#     * gradient clipping
# 

# ### LSTM cells constructor
# * Can construct and initialize multiple layers of any size
# * For regularization the keep rate in the dropout wrapper can be changed both for input and output

# In[10]:


def LSTM_cell(hidden_layer_size, batch_size, number_of_layers, dropout_rate_out = 1, dropout_rate_in = 1, dropout=True):
    
    layers = []
    i = 0
    while(i < number_of_layers):
        layers.append(tf.contrib.rnn.BasicLSTMCell(hidden_layer_size))
        if dropout:
            layers[i] = tf.contrib.rnn.DropoutWrapper(layers[i], 
                                                      output_keep_prob=dropout_rate_out, 
                                                      input_keep_prob=dropout_rate_in)
        i += 1
        
    cell = tf.contrib.rnn.MultiRNNCell(layers)
    
    init_state = cell.zero_state(batch_size, tf.float32)
    
    return cell, init_state


# ### Output layer; Losses and Optimizer

# In[11]:


def outputlayer_opt_loss(lstm_output, in_size, out_size, targets, learning_rate, grad_clip_margin, l1_l, l2_l, l1l2_l1, l1l2_l2, reg_type = 'no_penalty'):
    
    #Regularization
        #if error
    reg_types = ['no_penalty', 'L1', 'L2', 'L1L2']
    if reg_type not in reg_types:
        raise ValueError("Invalid reg type. Expected one of: %s" % reg_types)
    
        #define regularization type
    if reg_type == 'L1':
        regularizer = tf.contrib.layers.l1_regularizer(scale = l1_l)
    elif reg_type == 'L2':
        regularizer = tf.contrib.layers.l2_regularizer(scale = l2_l)
    elif reg_type == 'L1L2':
        regularizer = tf.contrib.layers.l1_l2_regularizer(scale_l1 = l1l2_l1, scale_l2 = l1l2_l2)
    
    
    #Output layer construction
    x = lstm_output[:, -1, :]
    print(x)
    weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.05), 
                          name='output_layer_weights') 
    if reg_type != 'no_penalty':
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weights) #for regularization
    
    bias = tf.Variable(tf.zeros([out_size]), name='output_layer_bias')
    
    output = tf.matmul(x, weights) + bias #output
    
    #Define losses
    losses = []
    for i in range(targets.get_shape()[0]):
        losses.append([(tf.pow(output[i] - targets[i], 2))])
    
    #Adjust according to regularization type
    if reg_type == 'no_penalty':   
        loss = tf.reduce_mean(losses)
    else:
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
        loss = tf.reduce_mean(losses) + reg_term
    
    #Clipping the gradient loss
    gradients = tf.gradients(loss, tf.trainable_variables())
    clipper_, _ = tf.clip_by_global_norm(gradients, grad_clip_margin)
    
    #Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
    
    return output, loss, train_optimizer


# ### Save Output

# In[12]:


from IPython.display import display,Javascript 
def save_notebook():
    return display(Javascript("IPython.notebook.save_notebook()"),
                   include=['application/javascript'])

def output_HTML(read_file, output_file):
    from nbconvert import HTMLExporter
    import codecs
    import nbformat
    exporter = HTMLExporter()
    # read_file is '.ipynb', output_file is '.html'
    output_notebook = nbformat.read(read_file, as_version=4)
    output, resources = exporter.from_notebook_node(output_notebook)
    codecs.open(output_file, 'w', encoding='utf-8').write(output)


# ### Multiple types of hyperparameters and regularization methods for crossvalidation
# 
# The parameters can be changed according to what settings are requested.

# In[13]:


#Hyperparameters
hidden_num = [1, 1, 2, 4]
hidden_size = [8, 16, 16, 48]
batch_size = 11
epoch = [800, 800, 800, 1500]

#Regularization
out_keep = [1, 1, 1, 1, 1, 1]
in_keep = [1, 1, 1, 1, 1, 1]
regs = ['L1', 'L1','L2', 'L2', 'L1L2', 'L1L2']
l1_lambdas = [0.001, 0.002, 0, 0, 0.0005, 0.0008]
l2_lambdas = [0, 0, 0.001, 0.002, 0.0005, 0.0008]
gradient_clip_mrg = 4 #To prevent exploding gradient, we use clipper to clip gradients below -margin or above this margin


# The below loops can be changed according to what combinations we wish to explore. 
# 
# * **First**: the outer loop were the number of layers, the inner loop was the layer size (Model ID and initializer need to be adjusted accordingly)
# * **Second**:, the outer layer ran through the 4 different models - chosen from the first run - the inner layer runs on the different dropouts (dropout to be modified above)
# * **Third**: same as second, except for the regularization changing to weight penalty

# In[23]:


MSE_train_all = []
MSE_test_all = []
Times = []
for l in range(0,len(hidden_num)):
    for m in range(0, len(regs)):
        epochs = epoch[l]
        ModelID = "NL"+str(hidden_num[l])+"_SL"+str(hidden_size[l])+"_DRI"+str(in_keep[m])+"_P"+str(regs[m])+"_L1Lam"+str(l1_lambdas[m])+"_L2Lam"+str(l2_lambdas[m])+"_EP"+str(epochs)
        print("Model ", ModelID, " starts running.")
            
        #Building the model:
        class StockPredictionRNN(object):

            def __init__(self, learning_rate=0.001, batch_size=batch_size, hidden_layer_size=hidden_size[l], number_of_layers=hidden_num[l], 
                         dropout=True, dropout_rate_out=out_keep[m] ,dropout_rate_in=in_keep[m], number_of_classes=1, 
                         gradient_clip_margin=gradient_clip_mrg, window_size=window_size, 
                         l1_l = l1_lambdas[m], l2_l = l2_lambdas[m], l1l2_l1 = l1_lambdas[m], l1l2_l2 = l2_lambdas[m], reg_type = regs[m]):

                self.inputs = tf.placeholder(tf.float32, [batch_size, window_size, 1], name='input_data')
                self.targets = tf.placeholder(tf.float32, [batch_size, 1], name='targets')

                cell, init_state = LSTM_cell(hidden_layer_size, batch_size, number_of_layers, dropout, dropout_rate_out, dropout_rate_in)

                outputs, states = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=init_state)

                self.outputs, self.loss, self.opt = outputlayer_opt_loss(outputs, hidden_layer_size, number_of_classes, self.targets, learning_rate, gradient_clip_margin, l1_l, l2_l, l1l2_l1, l1l2_l2, reg_type)
        tf.reset_default_graph() #resets graph in case there was one before
        model = StockPredictionRNN()
            
        ##TRAINING THE NETWORK:
        session =  tf.Session()
        tf.summary.FileWriter('./OUT', session.graph)
        session.run(tf.global_variables_initializer())
            
        tstart = time.time()
        
        mean_epoch_loss = []
        
        for i in range(epochs):
            traind_scores = []
            ii = 0
            epoch_loss = []
            while(ii + batch_size) <= len(X_train):
                X_batch = X_train[ii:ii+batch_size]
                y_batch = y_train[ii:ii+batch_size]

                o, c, _ = session.run([model.outputs, model.loss, model.opt], feed_dict={model.inputs:X_batch, model.targets:y_batch})

                epoch_loss.append(c)
                traind_scores.append(o)
                ii += batch_size
            if (i % 50) == 0:
                print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))
            mean_epoch_loss.append(np.mean(epoch_loss))
        
        Times.append((time.time()-tstart)/3600)
        print('Training ran for: ', (time.time()-tstart)/3600, ' hours. ModelID: ', ModelID)
            
        ##MEAN LOSSES PER EPOCHS
        plt.figure(figsize=(16, 7))
        plt.plot(mean_epoch_loss, label='Losses')
        plt.title('Mean Loss per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('./OUT/' +ModelID +'_LossPerEpoch.png', format='png', dpi=1000)
        plt.show()
            
        ## Calculating predictions
        ### train
        sup =[]
        for i in range(window_size):
            sup.append(None)
        for i in range(len(traind_scores)):
            for j in range(len(traind_scores[i])):
                sup.append(traind_scores[i][j])
        ### test
        tests = []
        i = 0
        while i + batch_size <= len(X_test):

            o = session.run([model.outputs], 
                            feed_dict={
                                model.inputs:X_test[i:i + batch_size]
                            })
            i += batch_size
            tests.append(o)
            
        tests_new = []
        for i in range(len(tests)):
            for j in range(len(tests[i][0])):
                tests_new.append(tests[i][0][j])
            
        test_results = []
        for i in range(len(data_new)-len(tests_new)):
            test_results.append(None)
        for i in range(len(tests_new)):
            test_results.append(tests_new[i])
            
        ### PLOTTING RESULTS
        plt.figure(figsize=(16, 7))
        plt.plot(train, label='Scaled training data')
        plt.plot(sup, label='Predicted training data')
        plt.legend()
        plt.savefig('./OUT/' + ModelID+'_PRED_train.png', format='png', dpi=1000)
        plt.show()

        plt.figure(figsize=(16, 7))
        plt.plot(test, label='Scaled test data')
        plt.plot(test_results[train_end:], label='Predicted test data')
        plt.legend()
        plt.savefig('./OUT/' + ModelID + '_PRED_test.png', format='png', dpi=1000)
        plt.show()
         
        plt.figure(figsize=(16, 7))
        plt.plot(data_new, label='Original data')
        plt.plot(sup, label='Training data')
        plt.plot(test_results, label='Testing data')
        plt.legend()
        plt.savefig('./OUT/' + ModelID + '_PRED_full.png', format='png', dpi=1000)
        plt.show()    
    
        MSE_test = 0
        MSE_train = 0

        #test
        for n in range(0,len(tests_new)):
            MSE_test += (tests_new[n]-test[n+window_size])**2

        #train
        for n in range(window_size, len(train)):
            MSE_train += (sup[n]-train[n])**2

        MSE_test = MSE_test/len(tests_new) 
        MSE_train = MSE_train/(len(train)-window_size)
        
        MSE_train_all.append(MSE_train)
        MSE_test_all.append(MSE_test)
        print("The training error is: ", MSE_train, ". The testing error is: ", MSE_test, ".")
            
        #Close session
        session.close()
            
#Save html
save_notebook()
time.sleep(3)
current_file = 'TimeSeries Forecast with LSTM-Loop_New-Step2_V3.ipynb'
output_file = './OUT/output_file_'+ str(time.strftime("%Y%m%d_%H_%M", time.gmtime())) + '.html' 
output_HTML(current_file, output_file)
            


# The below is to easily copy the errors and runtimes to excel:

# In[24]:


import win32clipboard as clipboard

def toClipboardForExcel(array):
    """
    Copies an array into a string format acceptable by Excel.
    Columns separated by \t, rows separated by \n
    """
    # Create string from array
    line_strings = []
    for line in array:
        line_strings.append("\t".join(line.astype(str)).replace("\n",""))
    array_string = "\r\n".join(line_strings)

    # Put string into clipboard (open, clear, set, close)
    clipboard.OpenClipboard()
    clipboard.EmptyClipboard()
    clipboard.SetClipboardText(array_string)
    clipboard.CloseClipboard()


# In[ ]:


toClipboardForExcel(MSE_train_all)

