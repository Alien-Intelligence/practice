import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,OneHotEncoder
import numpy as np
import random

a = pd.read_csv('voice_data.csv')

columns = ['meanfreq','sd','median','Q25','Q75','IQR','skew','kurt','sp.ent','sfm','mode','centroid',\
           'meanfun','minfun','maxfun','meandom','mindom','maxdom','dfrange','modindx','label']
company_lbl = LabelEncoder()
columnsToEncode = ['label']
for feature in columnsToEncode:
    a[feature] = company_lbl.fit_transform(a[feature])
li=[]
for feature in columns:
    if feature in columnsToEncode:
        li.append(pd.get_dummies(a[feature]))
    else:
        li.append(a[feature])
merged=pd.concat(li, axis=1).as_matrix()
train_x=merged[:2850, :-2]
train_y=merged[:2850, -2:]
test_x=merged[2850:, :-2]
test_y=merged[2850:, -2:]
def xavier_init(n_inputs, n_outputs, uniform =True):
    if uniform:
        init_range = tf.sqrt(6.0/(n_inputs+n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0/(n_inputs+n_outputs))
        return tf.truncated_normal_initializer(stddev = stddev)


nb_classes = 2

X = tf.placeholder(tf.float32, [None,20])
Y = tf.placeholder(tf.float32, [None, nb_classes])
dropout_rate = tf.placeholder("float")

W1 = tf.get_variable("W1", shape = [20,256], initializer = xavier_init(20,256))
W2 = tf.get_variable("W2", shape = [256,256], initializer = xavier_init(256,256))
W3 = tf.get_variable("W3", shape = [256,256], initializer = xavier_init(256,256))
W4 = tf.get_variable("W4", shape = [256,nb_classes], initializer = xavier_init(256,nb_classes))
b1 = tf.Variable(tf.random_normal([256]))
b2 = tf.Variable(tf.random_normal([256]))
b3 = tf.Variable(tf.random_normal([256]))
b4 = tf.Variable(tf.random_normal([nb_classes]))

_L1 = tf.nn.relu(tf.add(tf.matmul(X,W1),b1))
L1 = tf.nn.dropout(_L1, dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1,W2),b2))
L2 = tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2,W3),b3))
L3 = tf.nn.dropout(_L2, dropout_rate)

hypothesis = tf.add(tf.matmul(L3,W4),b4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = 0.004).minimize(cost)
is_correct = tf.equal(tf.arg_max(hypothesis,1),tf.arg_max(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
epoch = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        c, _ = sess.run([cost,optimizer], feed_dict = {X:train_x,Y:train_y,dropout_rate : 1.0})
        if i % int((epoch / 20)) == 0:
            print(i, "Cost:", c, "\n")
    print("Accuracy:", accuracy.eval(session = sess, feed_dict = {X: test_x, Y:test_y,dropout_rate : 1.0}))
    r = random.randint(0,250)
    print("Label :", sess.run(tf.argmax(test_y[r:r+1],1)))
    print("Prediction:", sess.run(tf.argmax(hypothesis,1), feed_dict = {X:test_x[r:r+1],dropout_rate : 1.0}))

# epoch : 500 , 3단 = 65%
# epoch : 500 , 3단, Xavier = 93.3%
# epoch : 500 , 3단, Xavier, dropout = 89.3%
# epoch : 500 , 4단, Xavier, dropout = 96.22%
