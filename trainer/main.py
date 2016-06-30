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
    reader = tf.WholeFileReader()
    key, value = reader.read(filenames)
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
      sess.run(init_op)
# x = tf.placeholder(tf.float32, [None, 30000])

if __name__ == '__main__':
    get_all_shots()
    init_tensorflow()
