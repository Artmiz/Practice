'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 60000]) # mnist data image of shape 28*28=784 -->changed it to 60k for network example
y = tf.placeholder(tf.float32, [None, 8]) # 0-9 digits recognition => 10 classes --> changed it to 8 for the number of actions in network example
###################################################################################

macro_state = []
cell_one_state = []
cell_two_state = []
cell_three_state = []
user_in_macro = []
user_cell_one = []
user_cell_two = []
user_cell_three = []
change_in_macro = []
change_cell_one = []
change_cell_two = []
change_cell_three = []
a = []
s = []

with open('a.txt') as f: #action file
     for line in f:
         a1, a2, a3, a4 = line.split()
         a1 = float(a1)
         a2 = float(a2)
         a3 = float(a3)
         a4 = float(a4)
         macro_state.append(a1)
         cell_one_state.append(a2)
         cell_two_state.append(a3)
         cell_three_state.append(a4)
         
a.append(macro_state)
a.append(cell_one_state) 
a.append(cell_two_state)
a.append(cell_three_state)

with open('s.txt') as g: #input file
     for line in g:
         s1, s2, s3, s4, s5, s6, s7, s8 = line.split()
         s1 = float(s1)
         s2 = float(s2)
         s3 = float(s3)
         s4 = float(s4)
         s5 = float(s5)
         s6 = float(s6)
         s7 = float(s7)
         s8 = float(s8)
         

         user_in_macro.append(s1)
         change_in_macro.append(s2)
         user_cell_one.append(s3)
         change_cell_one.append(s4)
         user_cell_two.append(s5)
         change_cell_two.append(s6)
         user_cell_three.append(s7)
         change_cell_three.append(s8)

s.append(user_in_macro)
s.append(change_in_macro)
s.append(user_cell_one)
s.append(change_cell_one)
s.append(user_cell_two)
s.append(change_cell_two)
s.append(user_cell_three)
s.append(change_cell_three)

x = a 
y = s 

###################################################################################
# Set model weights
W = tf.Variable(tf.zeros([60000, 8]))   #changed the weight from 784 to 60k and the 10 to 8
b = tf.Variable(tf.zeros([8])) # changed the bias from 10 to 8

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
