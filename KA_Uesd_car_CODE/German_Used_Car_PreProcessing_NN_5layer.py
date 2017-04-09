import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
#a = pd.read_csv('autos_Indexing.csv', na_values='Null')
train_x_str = []
train_y_str = []
test_x_str = []
test_y_str = []
with open('autos_Indexing.csv', "r") as rf:
    for i in rf.readlines()[2:10001]:
        train_x_str.append(i.split(',')[:12])
with open('autos_Indexing.csv', "r") as rf:
    for i in rf.readlines()[2:10001]:
        train_y_str.append(i.split(',')[12:13])
with open('autos_Indexing.csv', "r") as rf:
    for i in rf.readlines()[10001:11001]:
        test_x_str.append(i.split(',')[:12])
with open('autos_Indexing.csv', "r") as rf:
    for i in rf.readlines()[10001:11001]:
        test_y_str.append(i.split(',')[12:13])

train_x = [list(map(lambda x: float(x),train_x_str[i])) for i in range(len(train_x_str))]
train_y = [list(map(lambda x: float(x),train_y_str[i])) for i in range(len(train_y_str))]
test_x = [list(map(lambda x: float(x),test_x_str[i])) for i in range(len(test_x_str))]
test_y = [list(map(lambda x: float(x),test_y_str[i])) for i in range(len(test_y_str))]

def xavier_init(n_inputs, n_outputs, uniform =True):
    if uniform:
        init_range = tf.sqrt(6.0/(n_inputs+n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0/(n_inputs+n_outputs))
        return tf.truncated_normal_initializer(stddev = stddev)

#MinMaxSclaer
scaler = MinMaxScaler(feature_range=(0,1))
data_x = scaler.fit_transform(train_x)
data_y = scaler.fit_transform(train_y)

#Placeholder ( X, Y, 및 dropout_rate)
X = tf.placeholder(tf.float32, shape =[None,12])
Y = tf.placeholder(tf.float32, shape = [None,1])
dropout_rate = tf.placeholder("float")

#킹갓 4단
W1 = tf.get_variable("W1", shape = [12,256], initializer = xavier_init(12,256))
W2 = tf.get_variable("W2", shape = [256,256], initializer = xavier_init(256,256))
W3 = tf.get_variable("W3", shape = [256,256], initializer = xavier_init(256,256))
W4 = tf.get_variable("W4", shape = [256,256], initializer = xavier_init(256,256))
W5 = tf.get_variable("W5", shape = [256,1], initializer = xavier_init(256,1))

b1 = tf.Variable(tf.random_normal([256]))
b2 = tf.Variable(tf.random_normal([256]))
b3 = tf.Variable(tf.random_normal([256]))
b4 = tf.Variable(tf.random_normal([256]))
b5 = tf.Variable(tf.random_normal([1]))

_L1 = tf.nn.relu(tf.add(tf.matmul(X,W1),b1))
L1 = tf.nn.dropout(_L1, dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1,W2),b2))
L2 = tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2,W3),b3))
L3 = tf.nn.dropout(_L3, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3,W4),b4))
L4 = tf.nn.dropout(_L4, dropout_rate)
hypothesis = tf.add(tf.matmul(L4,W5),b5)

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
epoch = 2000
for s in range(epoch):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],feed_dict = {X:data_x,Y:data_y,dropout_rate : 0.7})
    if s % int((epoch/10)) == 0:
        print (s, "Cost:",cost_val,"\n")
    if s % int(epoch/5) == 0:
        print ("price",sess.run(hypothesis,feed_dict = {X : test_x, dropout_rate : 1.0}))
