import tensorflow as tf
import numpy as np
import os
from utils import *
from net import *
import json
import time

dim = 32
z_dim = 128
im_size = 224

import argparse
cmd_parser = argparse.ArgumentParser(description="tlnet config")
cmd_parser.add_argument('-c', '--classid', metavar='classid', type=str, default="02828884", help='Decide to select which class to train')
cmd_parser.add_argument('-g', '--gpu', metavar='gpu', type=int, default=1, help='Decide to train on which gpu')
cmd_parser.add_argument('-i', '--imgpergpu', metavar='imgpergpu', type=int, default=10, help='IMG_PER_GPU')

args = cmd_parser.parse_args()

dev_id = "gpu:{}".format(args.gpu)
train_cls = args.classid
batch_size = args.imgpergpu
print(dev_id)
print(train_cls)

lr = tf.placeholder(shape=[], dtype=tf.float32)
#is_training = tf.placeholder(dtype=tf.bool)

batchit = lambda x: [x() for _ in range(batch_size)]
imgs = batchit(lambda:tf.placeholder(shape=[im_size, im_size, 3], dtype=tf.float32))
x = batchit(lambda:tf.placeholder(shape=[dim, dim, dim, 1], dtype=tf.float32))

net_config = {
    "z_dim": z_dim,
    "batch_size": batch_size,
    "lambda_z": 1,
    "lambda_x": 0.2
}

tlnet = TLNet(**net_config)
with tf.device(dev_id):
   z_2d, z_3d, y_hat_3d, y_hat_2d = tlnet(x , imgs, is_training=True)

loss, t_loss = tlnet.get_loss(x, z_2d, z_3d, y_hat_3d, y_hat_2d)

with tf.device(dev_id):
    optimiter = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt1 = optimiter.minimize(loss[1])
        opt2 = optimiter.minimize(loss[0])
        opt3 = optimiter.minimize(t_loss)

tf.summary.scalar("tl_net/z_loss", loss[0])
tf.summary.scalar("tl_net/x_loss", loss[1])
tf.summary.scalar("tl_net/total_loss", t_loss)
tf.summary.scalar("learning_rate", lr)
merged = tf.summary.merge_all()

with open("data/splits.json", "r") as f :
    data_dict = json.load(f)
data_list = data_dict[train_cls]['train']

data_config = {
     "DATA_DIR": "data/shapenet_lsm",
     "data_list": data_list,
     "train_cls": train_cls,
     "img_fixed_size": None
}

data_loader = DataSet(**data_config)
data_loader()

epoch = len(data_list)//batch_size
print("epoch=%d"%epoch)
max_itrs = 800*epoch
stage_1 = 200
stage_2 = 400
stage = 3
lrs = 445
learning_r = 0.0005

model_saver = tf.train.Saver()

saver = tf.train.Saver(
    write_version=tf.train.SaverDef.V1,
    var_list=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="resnet_v2_50")
)

opt = opt1

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with tf.Session(config=tf_config) as sess:
    train_writer = tf.summary.FileWriter("log/tl-net-{}.T1/z{}_x{}_stage{}_lr{}".format(train_cls,tlnet.lambda_z,tlnet.lambda_x,stage,lrs), sess.graph)
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, save_path="resnet_v2_50/resnet_v2_50.ckpt")
    start = time.time()
    for i in range( max_itrs):
        feed_dict=data_loader._feed_dict(batch_size, x, imgs)
        if(i == stage_1 * epoch):
            learning_r = 0.0005
            opt = opt2
        elif(i == stage_2 * epoch):
            learning_r = 0.00005
            opt = opt3
        feed_dict[lr] = learning_r
        _ = sess.run([opt], feed_dict)
        '''
        if(i == stage_1 * epoch):
            opt = opt2
        elif(i == stage_2 * epoch):
            opt = opt3
        '''
        if i % 20 == 0:
            print("\033[1;34mepoch: %d\n\033[1;33m<%d> \033[1;35mlearning rate: \033[1;31m%.5f\033[0m"%(i//epoch,i, learning_r))
            result = sess.run(loss + (t_loss, merged), feed_dict=feed_dict)
            summary = result[-1]
            print("z_loss:\t%.8f\nx_loss:\t%.8f\n-----\n\033[1;32mTOTAL LOSS:\t\033[1;33m%.8f\033[0m\n"%tuple(result[:-1]))
            train_writer.add_summary(summary, i)
        if i % 5000 == 0 and  i >0:
            model_saver.save(sess, "models-{}-z{}_x{}_stage{}_lr{}/tl-net-{}.v1".format(train_cls,tlnet.lambda_z,tlnet.lambda_x,stage,lrs,train_cls), global_step=i)
    model_saver.save(sess, "models-{}-z{}_x{}_stage{}_lr{}/tl-net-{}.v1".format(train_cls,tlnet.lambda_z,tlnet.lambda_x,stage,lrs,train_cls), global_step=i)
    end = time.time()
    use_time = end - start
    print("\033[1;36mtime\033[1;35m%.2fs\n\033[1;36mspeed\033[1;33m%.4f sample/s\033[0m"%(use_time, max_itrs*batch_size/use_time))
