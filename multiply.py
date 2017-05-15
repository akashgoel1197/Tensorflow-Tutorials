import tensorflow as tf

a = tf.placeholder("float32")
b = tf.placeholder("float32")

y = tf.multiply(a,b)

with tf.Session() as sess:
    print("Hadmard product ....")
    print(sess.run(y,feed_dict ={ a:[[2,4],[4,3]], b: [[3,4],[2,5]]}))
    
                                            
    
#FOR VECTOR MULTIPLICATION

z = tf.matmul(a,b)

with tf.Session() as sess:
    print(" matrix product matmul ...")
    print(sess.run(z,feed_dict={a:[[2,4],[4,3]],b:[[3,4],[2,5]]}))
    
