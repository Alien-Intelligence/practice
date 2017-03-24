import tensorflow as tf
import numpy as np
xy = np.loadtxt('kbo2016_1.csv', delimiter = ',', dtype = np.float32)
x_data =xy[:-1,1:]
y_data =xy[:-1,[0]]
x_test = xy[[-1],1:]
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

X = tf.placeholder(tf.float32, shape =[None,4])
Y = tf.placeholder(tf.float32, shape = [None,1])

w = tf.Variable(tf.random_normal([4,1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.matmul(X,w) +b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0000001)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for s in range(20000):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],feed_dict = {X:x_data,Y:y_data})
    if s % 5000 == 0:
        print (s, "Cost:",cost_val,"\nPrediction\n",hy_val)
        print ("KT's run will be",sess.run(hypothesis,feed_dict = {X : x_test}))
