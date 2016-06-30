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
    init_op = tf.initialize_all_variables()
    # with tf.Session() as sess:
    #   sess.run(init_op)
    images = tf.convert_to_tensor(ALL_FILENAMES)
    labels = tf.convert_to_tensor(ALL_VECTORS)
# x = tf.placeholder(tf.float32, [None, 30000])
    input_queue = tf.train.slice_input_producer([images, labels])
    print(input_queue)

    #image, label = read_images_from_disk(input_queue)

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
      sess.run(init_op)
      tf.train.start_queue_runners(sess=sess)
    #   sess.run([images[0], labels[0]])

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
