import tensorflow as tf
import trainer
import numpy as np
import os

IMAGE_WIDTH = 40
IMAGE_HEIGHT = 30
IMAGE_DEPTH = 3
TOTAL_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH
LABEL_SIZE = 19
BATCH_SIZE = 128
TRAIN_DIR = "tmp/ai_design_train"
# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')

def get_all_shots():
    all_files, all_vectors = trainer.get_all_shots()
    run_training(all_vectors)

def run_training(all_vectors):

    all_vectors = np.asarray(all_vectors)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        # Input images and labels.
        images, labels = trainer.inputs(train=True, batch_size=BATCH_SIZE,
                                num_epochs=2)

        print(images)

        x = tf.placeholder(tf.float32, [None, TOTAL_PIXELS])

        W = tf.Variable(tf.zeros([TOTAL_PIXELS, LABEL_SIZE]))
        b = tf.Variable(tf.zeros([LABEL_SIZE]))

        y = tf.nn.softmax(tf.matmul(x, W) + b)

        y_ = tf.placeholder(tf.float32, [None, LABEL_SIZE])

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        vector_initializer = tf.placeholder(dtype=all_vectors.dtype, shape=all_vectors.shape)

        init = tf.initialize_all_variables()

        sess = tf.Session()
        sess.run(init)

        vector_inputs = tf.Variable(vector_initializer, trainable=False, collections=[])
        sess.run(vector_inputs.initializer, feed_dict={vector_initializer:all_vectors})

        graph = tf.get_default_graph()
        #print(graph.get_operations())
        print(tf.shape(images))
        print(vector_inputs)

        ims = tf.reshape(images, [len(all_vectors), TOTAL_PIXELS])
        saver = tf.train.Saver(tf.all_variables())
        with sess.as_default():
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(100):
                print(i)
                sess.run(train_step, feed_dict={x:ims.eval(), y_: vector_inputs.eval()})
            coord.request_stop()
            coord.join(threads)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(accuracy.eval({x: ims.eval(), y_: vector_inputs.eval()}))
            classification = sess.run(tf.argmax(y, 1), feed_dict={x:[ims.eval()[0]]})
            print 'Neural Network predicted', classification
            #print(y_.eval())
        #tf.app.run()

def image_data(all_filenames):
    pass

def main(_):
    pass


if __name__ == '__main__':
    get_all_shots()
    # run_training()
