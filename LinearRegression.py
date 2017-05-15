import tensorflow as tf
import numpy as np

# DataSet

trX = np.linspace(-1,1,100)
trY = 2*trX +np.random.randn(*trX.shape)*0.33

#Model

X = tf.placeholder("float")
Y = tf.placeholder("float")

def model(X,w,b):
    return tf.multiply(X,w) + b

w = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0 , name = "bias")

y_model = model(X,w,b)

cost = tf.square(Y-y_model)

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        for (x,y) in zip(trX,trY):
            sess.run(train_op,feed_dict ={X:x,Y:y})
        print (i+1)
        print(sess.run(w))
        print (sess.run(b))
