import tensorflow as tf
import numpy as np
import os
from utils import *
from net import *
import json
import time

def write_voxel_in_file(f, voxel, threshold):
    xx , yy , zz = voxel.shape[:3]
    for ii in range(xx):
        for jj in range(yy):
            for kk in  range(zz):
                if voxel[ii,jj,kk,0]>threshold:
                    f.write("{} {} {} {}\n".format(ii,jj,kk,voxel[ii,jj,kk,0]))

dim = 32
batch_size = 20
z_dim = 128
im_size = 224

import argparse
cmd_parser = argparse.ArgumentParser(description="tlnet config")
cmd_parser.add_argument('-c', '--classid', metavar='classid', type=str, default="02828884", help='Decide to select which class to train')
cmd_parser.add_argument('-g', '--gpu', metavar='gpu', type=int, default=1, help='Decide to train on which gpu')

args = cmd_parser.parse_args()

dev_id = "gpu:{}".format(args.gpu)
train_cls = args.classid
model_version = "-z1_x0.2_stage1_lr445"

lr = tf.placeholder(shape=[], dtype=tf.float32)

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
   z_2d, z_3d, y_hat_3d, y_hat_2d = tlnet(x , imgs, is_training=False)

loss, t_loss = tlnet.get_loss(x, z_2d, z_3d, y_hat_3d, y_hat_2d)

ysig_3d = tf.sigmoid(y_hat_3d)
ysig_2d = tf.sigmoid(y_hat_2d)

with open("data/splits.json", "r") as f :
    data_dict = json.load(f)
data_list = data_dict[train_cls]['test']

DATA_DIR = "data/shapenet_lsm"

batch_list = np.random.choice(len(data_list), batch_size)
ri_list = []

feed_dict = {}
for i in range(batch_size):
    voxel_path = os.path.join(DATA_DIR,"voxels","modelVoxels"+str(32),train_cls,data_list[batch_list[i]]+".mat")
    render_path = os.path.join(DATA_DIR,"renders",train_cls,data_list[batch_list[i]])
    try:
        os.makedirs("test_out_{}/{}".format(model_version,data_list[i]))
    except:
        pass
    voxel = np.expand_dims(scio.loadmat(voxel_path)['Volume'],3)
    f = open("test_out_{}/{}.xyz".format(model_version,data_list[batch_list[i]]), "w+")
    write_voxel_in_file(f, voxel, 0.5)
    f.close()

    feed_dict[x[i]] = voxel
    ri = random.randint(0, 19)
    ri_list.append(ri)
    img = cv2.imread(os.path.join(render_path,"render_{}.png".format(ri)))
    cv2.imwrite('test_out_{}/{}_render_{}.png'.format(model_version,data_list[batch_list[i]],ri),img)
    feed_dict[imgs[i]] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


model_saver = tf.train.Saver()

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with tf.Session(config=tf_config) as sess:
    module_file = tf.train.latest_checkpoint("models-{}".format(train_cls)+model_version)
    model_saver.restore(sess, module_file)
    result = sess.run(loss + (t_loss,z_2d,z_3d, ysig_3d, ysig_2d), feed_dict=feed_dict)
    print("z_loss:\t%.8f\nx_loss:\t%.8f\n-----\n\033[1;32mTOTAL LOSS:\t\033[1;33m%.8f\033[0m\n"%tuple(result[:-4]))
    rst_2d = result[-1]
    rst_3d = result[-2]
    z_g_3d = result[-3]
    z_g_2d = result[-4]

f = open("{}_z_3d".format(train_cls), "w+")
for i in range(batch_size):
    for j in range(z_dim):
        f.write("{:.4f},".format(z_g_3d[i,0,0,0,j]))
    f.write("\n")
f.close()
f = open("{}_z_2d".format(train_cls), "w+")
for i in range(batch_size):
    for j in range(z_dim):
        f.write("{:.4f},".format(z_g_2d[i,0,0,0,j]))
    f.write("\n")
f.close()

threshold = [0.1, 0.3, 0.5, 0.7, 0.9]
data_list[batch_list[i]]
for i in range(batch_size):
    for j in threshold:
        f = open("test_out_{}/{}_{}_{}_{}.xyz".format(model_version,data_list[batch_list[i]],"2d",ri_list[i],j), "w+")
        write_voxel_in_file(f, rst_2d[i], j)
        f.close()
        f = open("test_out_{}/{}_{}_{}_{}.xyz".format(model_version,data_list[batch_list[i]],"3d",ri_list[i],j), "w+")
        write_voxel_in_file(f, rst_3d[i], j)
        f.close()
