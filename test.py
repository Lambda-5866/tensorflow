import tensorflow.compat.v1 as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape([-1, 784])
test_images  = test_images.reshape([-1,784])
train_images = train_images / 255.
test_images  = test_images / 255.
print(train_images[0])
#train_labels = train_labels.reshape([-1, 784])
print('train_images.shape : ', train_images.shape)
#from tensorflow.examples.tutorials.mnist import input_data

tf.disable_v2_behavior()

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

#batch_xs, batch_ys = mnist.train.next_batch(100)

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

prediction = tf.argmax(hypothesis, 1) #예측한 결과를 0~6사이의 값으로 만든다
is_correct = tf.equal(prediction, tf.argmax(Y, 1))#예측한 결과와 Y 데이터를 비교
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) #이것들을 평균낸다

training_epochs = 15
batch_size = 100

import matplotlib.pyplot as plt
import random

#'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(train_images.shape[0] / batch_size)

        for i in range(total_batch):
            s_idx = int(train_images.shape[0] * i / total_batch)
            e_idx = int(train_images.shape[0] * (i+1)/ total_batch)
            #print('s_idx : ', s_idx)
            #print('width : ', width)
            batch_xs = train_images[s_idx : e_idx]
            batch_ys = train_labels[s_idx : e_idx]
            #print('batch_xs.shape : ', batch_xs.shape)
            #print('batch_ys.shape : ', batch_ys.shape)
            #Y_one_hot = tf.one_hot(batch_ys, nb_classes)
            Y_one_hot = np.eye(nb_classes)[batch_ys]
            #print('Y_one_hot.shape :', Y_one_hot.shape)
            _,c = sess.run([optimizer, cost], feed_dict={X:batch_xs, Y:Y_one_hot})
            #print('total_batch : ', total_batch, ', c:', c)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost))
    Y_one_hot = np.eye(nb_classes)[test_labels]
    print("Accuracy : ", accuracy.eval(session=sess, feed_dict={X:test_images, Y:Y_one_hot}))
    
    r = random.randint(0, test_images.shape[0] - 1)
    print('label : ', test_labels[r:r+1])
    print('Prediction : ', sess.run(tf.argmax(hypothesis, 1), feed_dict={X:test_images[r:r+1]}))
    plt.imshow(test_images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    plt.show()
#'''
