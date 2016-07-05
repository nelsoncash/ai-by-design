import tensorflow as tf
import json

ALL_SHOTS = []
ALL_VECTORS = []
ALL_IDS = []
ALL_FILENAMES = []

def get_all_shots():
    with open('../db/db.json') as db_file:
        db_data = json.load(db_file)
        for k, v in db_data.iteritems():
            ALL_SHOTS.append(v)
            ALL_VECTORS.append(v['tagVector'])
            ALL_IDS.append(v['id'])
            ALL_FILENAMES.append("".join(("../tmp-bw/", k, ".png")))
        print(ALL_VECTORS)
        print(ALL_FILENAMES)

def init_tensorflow():
    filenames = tf.train.string_input_producer(ALL_FILENAMES)
    # reader = tf.WholeFileReader()
    # key, value = reader.read(filenames)
    images = tf.convert_to_tensor(ALL_FILENAMES)
    labels = tf.convert_to_tensor(ALL_VECTORS)
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
      sess.run(init_op)
    print(labels.shape[0])
    print(images)
# x = tf.placeholder(tf.float32, [None, 30000])
    # input_queue = tf.train.slice_input_producer([images, labels])
    # print(input_queue)
    #
    # #image, label = read_images_from_disk(input_queue)
    #
    # init_op = tf.initialize_all_variables()
    # graph = tf.get_default_graph()
    # for op in  graph.get_operations():
    #     print(op.get_attr())
    # print(operations)
    # with tf.Session() as sess:
    #   sess.run(init_op)
    #   tf.train.start_queue_runners(sess=sess)
    #   sess.run([images[0], labels[0]])

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([mnist.IMAGE_PIXELS])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label

def inputs(num_epochs):
    filename_queue = tf.train.string_input_producer(
        ALL_FILENAMES, num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=2,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000)

    return images, sparse_labels

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    print(file_contents)
    example = tf.image.decode_png(file_contents, channels=3)
    print(example)
    return example, label

if __name__ == '__main__':
    get_all_shots()
    init_tensorflow()
