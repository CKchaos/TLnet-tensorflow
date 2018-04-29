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
cmd_parser.add_argument('-g', '--gpu', metavar='gpu', type=str, default='0', help='num of gpu')
cmd_parser.add_argument('-i', '--imgpergpu', metavar='imgpergpu', type=int, default=10, help='IMG_PER_GPU')

args = cmd_parser.parse_args()

gpus = args.gpu
IMG_PER_GPU = args.imgpergpu
train_cls = args.classid
print(train_cls)

gpus = [int(gpus[i]) for i in range(len(gpus))]
NUM_GPU = len(gpus)

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [g for g, _ in grad_and_vars]
        grad = tf.stack(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

net_config = {
    "z_dim": z_dim,
    "batch_size": IMG_PER_GPU,
    "lambda_z": 1,
    "lambda_x": 0.2
}

def Model(reuse):
    batchit = lambda x: [x() for _ in range(IMG_PER_GPU)]
    imgs = batchit(lambda:tf.placeholder(shape=[im_size, im_size, 3], dtype=tf.float32))
    x = batchit(lambda:tf.placeholder(shape=[dim, dim, dim, 1], dtype=tf.float32))

    tlnet = TLNet(**net_config)
    z_2d, z_3d, y_hat_3d, y_hat_2d = tlnet(x , imgs, True,reuse)

    loss, t_loss = tlnet.get_loss(x, z_2d, z_3d, y_hat_3d, y_hat_2d)

    return imgs, x, loss[0], loss[1], t_loss


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with tf.Session(config=tf_config) as sess:
    with tf.device('/cpu:0'):
        lr = tf.placeholder(shape=[], dtype=tf.float32)
        optimiter = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
        print('build model...')
        print('build model on gpu tower...')
        models = []
        grads = []
        for gpu_id in gpus:
            with tf.device('/gpu:%d' % gpu_id):
                print('tower:%d...'% gpu_id)
                with tf.name_scope('tower_%d' % gpu_id):
                    imgs, x, loss_z, loss_x, t_loss  = Model(gpu_id>0)
                    tvars = tf.trainable_variables()
                    varsx = [var for var in tvars if(('e_' in var.name) or ('g_' in var.name))]
                    print(len(varsx))
                    varsz = [var for var in tvars if ('e_' not in var.name) and ('g_' not in var.name)]
                    print(len(varsz))
                    grad = optimiter.compute_gradients(t_loss)
                    gradx = optimiter.compute_gradients(loss_x,varsx)
                    gradz = optimiter.compute_gradients(loss_z,varsz)
                    models.append((imgs, x, loss_z, loss_x, t_loss, grad, gradz, gradx))
                    #grads.append(grad)
        print('build model on gpu tower done.')

        print('reduce model on cpu...')
        _, _, tower_loss_z, tower_loss_x, tower_t_loss, tower_grads, tower_grads_z, tower_grads_x = zip(*models)
        #_, _, tower_losses, tower_losses_without_cls, tower_d_pos, tower_d_neg = zip(*models)
        aver_t_loss_op = tf.reduce_mean(tower_t_loss)
        aver_loss_x_op = tf.reduce_mean(tower_loss_x)
        aver_loss_z_op = tf.reduce_mean(tower_loss_z)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            apply_gradient_op = optimiter.apply_gradients(average_gradients(tower_grads))
            apply_gradient_op_z = optimiter.apply_gradients(average_gradients(tower_grads_z))
            apply_gradient_op_x = optimiter.apply_gradients(average_gradients(tower_grads_x))

        tf.summary.scalar("tl_net/z_loss", aver_loss_z_op)
        tf.summary.scalar("tl_net/x_loss", aver_loss_x_op)
        tf.summary.scalar("tl_net/total_loss", aver_t_loss_op)
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

        epoch = len(data_list)//(NUM_GPU*IMG_PER_GPU)
        print("epoch=%d"%epoch)
        max_itrs = 1000*epoch
        stage_1 = 250
        stage_2 = 500
        stage = 3
        lrs = 445
        learning_r = 0.0005
        opt = apply_gradient_op_x

        model_saver = tf.train.Saver()

        saver = tf.train.Saver(
            write_version=tf.train.SaverDef.V1,
            var_list=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope="resnet_v2_50")
        )

        train_writer = tf.summary.FileWriter("log/tl-net-{}.T1/z{}_x{}_stage{}_lr{}".format(train_cls,net_config['lambda_z'],net_config['lambda_x'],stage,lrs), sess.graph)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, save_path="resnet_v2_50/resnet_v2_50.ckpt")
        start = time.time()
        for i in range( max_itrs):
            feed_dict=data_loader._feed_all_gpu(models,IMG_PER_GPU)
            if(i == stage_1 * epoch):
                learning_r = 0.0005
                opt = apply_gradient_op_z
            elif(i == stage_2 * epoch):
                learning_r = 0.00005
                opt = apply_gradient_op
            feed_dict[lr] = learning_r
            _ = sess.run([opt], feed_dict)

            if i % 20 == 0:
                print("\033[1;34mepoch: %d\n\033[1;33m<%d> \033[1;35mlearning rate: \033[1;31m%.5f\033[0m"%(i//epoch,i, learning_r))
                result = sess.run((aver_loss_z_op, aver_loss_x_op, aver_t_loss_op, merged), feed_dict=feed_dict)
                summary = result[-1]
                print("z_loss:\t%.8f\nx_loss:\t%.8f\n-----\n\033[1;32mTOTAL LOSS:\t\033[1;33m%.8f\033[0m\n"%tuple(result[:-1]))
                train_writer.add_summary(summary, i)
            if i % 5000 == 0 and  i >0:
                model_saver.save(sess, "models-{}-z{}_x{}_stage{}_lr{}/tl-net-{}.v1".format(train_cls,net_config['lambda_z'],net_config['lambda_x'],stage,lrs,train_cls), global_step=i)
        model_saver.save(sess, "models-{}-z{}_x{}_stage{}_lr{}/tl-net-{}.v1".format(train_cls,net_config['lambda_z'],net_config['lambda_x'],stage,lrs,train_cls), global_step=i)
        end = time.time()
        use_time = end - start
        print("\033[1;36mtime\033[1;35m%.2fs\n\033[1;36mspeed\033[1;33m%.4f sample/s\033[0m"%(use_time, max_itrs*(NUM_GPU*IMG_PER_GPU)/use_time))
