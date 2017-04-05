import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
train_x_str = []
train_y_str = []
test_x_str = []
test_y_str = []
with open('Train_major.csv', "r") as rf:
    for i in rf.readlines():
        train_x_str.append(i.split(',')[3:5]+i.split(',')[5:26])
with open('Train_major.csv', "r") as rf:
    for i in rf.readlines():
        train_y_str.append(i.split(',')[5:6])
with open('Test_major.csv', "r") as rf:
    for i in rf.readlines():
        test_x_str.append(i.split(',')[3:5]+i.split(',')[5:26])
with open('Test_major.csv', "r") as rf:
    for i in rf.readlines():
        test_y_str.append(i.split(',')[5:6])

def xavier_init(n_inputs, n_outputs, uniform =True):
    if uniform:
        init_range = tf.sqrt(6.0/(n_inputs+n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0/(n_inputs+n_outputs))
        return tf.truncated_normal_initializer(stddev = stddev)

train_x = [list(map(lambda x: float(x),train_x_str[i])) for i in range(len(train_x_str))]
train_y = [list(map(lambda x: float(x),train_y_str[i])) for i in range(len(train_y_str))]
test_x = [list(map(lambda x: float(x),test_x_str[i])) for i in range(len(test_x_str))]
test_y = [list(map(lambda x: float(x),test_y_str[i])) for i in range(len(test_y_str))]

scaler = MinMaxScaler(feature_range=(0,100))
data_x = scaler.fit_transform(train_x)
data_y = scaler.fit_transform(train_y)
X = tf.placeholder(tf.float32, shape =[None,23])
Y = tf.placeholder(tf.float32, shape = [None,1])

w = tf.get_variable("W1", shape = [23,1], initializer = xavier_init(23,1))
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.matmul(X,w) +b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.000005)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for s in range(200000):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],feed_dict = {X:data_x,Y:data_y})
    if s % 50000 == 0:
        print (s, "Cost:",cost_val,"\nPrediction\n",hy_val)
        print ("ssss",sess.run(hypothesis,feed_dict = {X : test_x}))
