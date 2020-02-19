import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from lenet import Lenet
import matplotlib.pyplot as plt
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants, signature_def_utils, tag_constants, utils)

# data for train and test
mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)
mnist1 = input_data.read_data_sets('./MNIST_data/')

# padding mnist 28x28->32x32
def mnist_reshape_32(_batch):
    batch = np.reshape(_batch,[-1,28,28,1])
    batch_32 = np.pad(batch, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=0)
    return batch_32

# read special num frpm mnist
def mnist_read_num(_num,_batch):
    labels_num = mnist1.test.labels
    num = _num * np.ones(labels_num.shape)
    mask = np.equal(labels_num, num)
    image = mnist1.test.images[mask, :]
    return image[0:_batch]

# create labels in form of [size , 10]
def mnist_label_maker(_num,_batch):
    labels = np.zeros([_batch,10])
    labels[:,_num] = 1
    return labels

def export_model(session, m):
   #只需要修改这一段，定义输入输出，其他保持默认即可
    model_signature = signature_def_utils.build_signature_def(
        inputs={"input": utils.build_tensor_info(m.a)},
        outputs={
            "output": utils.build_tensor_info(m.y)},

        method_name=signature_constants.PREDICT_METHOD_NAME)

    export_path = "./lenet/lenet_pb"
    if os.path.exists(export_path):
        os.system("rm -rf "+ export_path)
    print("Export the model to {}".format(export_path))

    try:
        legacy_init_op = tf.group(
            tf.tables_initializer(), name='legacy_init_op')
        builder = saved_model_builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            session, [tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    model_signature,
            },
            legacy_init_op=legacy_init_op)

        builder.save()
    except Exception as e:
        print("Fail to export saved model, exception: {}".format(e))

#create net
net = Lenet()

#train
merged_summary = tf.summary.merge_all()
sess = tf.Session()
writer = tf.summary.FileWriter('./' + "lenet")#tensorboard --logdir=C:\Users\k\PycharmProjects\untitled\lenet
writer.add_graph(sess.graph)

init = tf.global_variables_initializer ()
sess.run(init)
step = 500
print_step = 50


#ckpt saver
# m_saver = tf.train.Saver()
# m_path = "./model/lenet.ckpt"

for i in range(step):
    batch_xs ,batch_ys = mnist.train.next_batch (100)

    sess.run([net.train_step], feed_dict={ net.x: mnist_reshape_32(batch_xs),  net.y_: batch_ys})

    # print
    if i % print_step == 0:
        print('[accuracy , loss]:',
              sess.run([net.accuracy , net.loss],
                       feed_dict={net.x: mnist_reshape_32(mnist.test.images),
                                  net.y_: mnist.test.labels}))

    # save
    if i % 499 == 0:
        # m_saver.save(sess, m_path, global_step = i)
        # print("Model saved")
        # 含有变量
        export_model(sess,net)
