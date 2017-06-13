#!/usr/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def get_weights(n_features, n_labels):
	return (tf.Variable(tf.truncated_normal(n_features,n_labels)))  # generate n_features*n_labels dimesions matrix

def get_bias(n_labels):
	return (tf.Variable(tf.zeros(n_labels)))

def linear(input, W, b):
	return (tf.add(tf.matmul(input,W),b))


tf.reset_default_graph()


learning_rate = 0.001
n_input = 784
n_classes = 10

# Import MNIST data
mnist = input_data.read_data_sets('.', one_hot=True)

# The features are already scaled and the data is shuffled
features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])



weights = tf.Variable(tf.truncated_normal([n_input,n_classes]))
biases = tf.Variable(tf.truncated_normal([n_classes]))

logits = tf.add(tf.matmul(features, weights), biases)

# loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

import math

save_file = "./train_model.ckpt"
batch_size = 128
n_epochs = 100

saver = tf.train.Saver()

# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	# Training cycle
# 	for epoch in range(n_epochs):
# 		total_batch = math.ceil(mnist.train.num_examples / batch_size)
# 		# Loop over all batches
# 		for i in range(total_batch):
# 			batch_features, batch_labels = mnist.train.next_batch(batch_size)
# 			sess.run(
# 				optimizer,
# 				feed_dict={features: batch_features, labels: batch_labels})

# 		# Print status for every 10 epochs
# 		if epoch % 10 == 0:
# 			valid_accuracy = sess.run(
# 				accuracy,
# 				feed_dict={
# 				features: mnist.validation.images,
# 				labels: mnist.validation.labels})
# 			print('Epoch {:<3} - Validation Accuracy: {}'.format(
# 				epoch,
# 				valid_accuracy))
# 	# Save the model
# 	saver.save(sess, save_file)
# 	print('Trained Model Saved.')

# Launch the graph
with tf.Session() as sess:
	saver.restore(sess, save_file)

	test_accuracy = sess.run(
		accuracy,
		feed_dict={features: mnist.test.images, labels: mnist.test.labels})

print('Test Accuracy: {}'.format(test_accuracy))











