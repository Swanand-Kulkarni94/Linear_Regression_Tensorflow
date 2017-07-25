#Program 5, Simple Linear Regression using Tensorflow

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

tf.set_random_seed(1)
np.random.seed(1)

#Generate dumy data
x = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, size = x.shape)
y = np.power(x, 2) + noise

#Plot data
plt.scatter(x, y)
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)
tf_y = tf.placeholder(tf.float32, y.shape)

#Neural Network Layer
layer1 = tf.layers.dense(tf_x, 10, tf.nn.relu)
output = tf.layers.dense(layer1, 1)

loss = tf.losses.mean_squared_error(tf_y, output)
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5) #Vary the values of Learning Rate, to learn about various rates
train_op = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()

for step in range(100):
	#Train and obtain an output
	_, l, pred = sess.run([train_op, loss, output], {tf_x: x, tf_y: y})
	if step % 5 == 0:
		#Plot and show learning process
		plt.cla()
		plt.scatter(x, y)
		plt.plot(x, pred, 'r-', lw = 5)
		plt.text(0.5, 0, 'Loss = %.4f' % l, fontdict = {'size': 20, 'color': 'red'})

plt.ioff()
plt.show()