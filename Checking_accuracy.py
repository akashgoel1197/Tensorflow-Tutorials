import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

alpha = 0.01
epochs = 25
batch_size = 100
display_step = 10

x = tf.placeholder(tf.float32,shape =[None,784])
y = tf.placeholder(tf.float32, shape=[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

func = tf.nn.softmax(tf.matmul(x,W) + b)
# func return tensor of size None*10
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(func),reduction_indices=1))

train = tf.train.GradientDescentOptimizer(alpha).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            xs,ys = mnist.train.next_batch(batch_size)
            _,c = sess.run([train,cost],feed_dict ={x:xs,y:ys})
            avg_cost += c/total_batch
        print(avg_cost)

    print('finished')

    correct_prediction = tf.equal(tf.argmax(func,1),tf.argmax(y,1))
    a = tf.shape(correct_prediction)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    print (accuracy.eval({x : mnist.test.images, y:mnist.test.labels}))
    print(a.eval({x : mnist.test.images, y:mnist.test.labels}))





    
