import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2
#from slim.nets import resnet_v2


class TLNet():

    def __init__(self, **kwargs):

        self.__dict__.update(kwargs)

    def __call__(self, x, imgs, is_training, reuse=False):

       z_2d = self._ConvNet2D(imgs, is_training, reuse=reuse)
       y_hat_2d = self._decoder(z_2d, is_training, reuse=reuse)
       return y_hat_2d

    def _ConvNet2D(self, x, is_training, reuse=False):

        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            f, _ = resnet_v2.resnet_v2_50(
                x,
                num_classes=None,
                is_training=is_training,
                global_pool=False,
                reuse=reuse)
            print("resnet.out.shape: %s"%f.get_shape())
            with tf.variable_scope("ConvNet2D", reuse=reuse):
                f = tf.reduce_mean(f, [1, 2], name='global_avg_pooling', keep_dims=True)
                z = slim.conv2d(f,
                    4096,
                    [1,1],
                    padding='VALID',
                    normalizer_fn=None, scope='f2zfeture')
                z = slim.conv2d(z,
                    self.z_dim,
                    [1,1],
                    padding='VALID',
                    normalizer_fn=None, scope='z_2d')

                #g_feature = tf.squeeze(g_feature, [1, 2], name='global_spatial_squeeze')

                return tf.expand_dims(z,1)

    def _decoder(self, x, is_training, reuse=False):

        with tf.variable_scope("decoder",reuse = reuse):

            w_conv1 = tf.get_variable('g_wconv0', [4, 4, 4, 64, self.z_dim], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv1 = tf.get_variable('g_bconv0', [64], initializer=tf.constant_initializer(.1))
            h_conv1 = tf.nn.conv3d_transpose(x, w_conv1, output_shape=[self.batch_size, 4, 4, 4, 64], strides=[1, 1, 1, 1, 1], padding='VALID')+b_conv1
            h_norm1 = tf.contrib.layers.batch_norm(inputs = h_conv1, center=True, scale=True, is_training=is_training, scope="g_bn1")
            h_relu1 = tf.nn.relu(h_norm1)

            w_conv2 = tf.get_variable('g_wconv1', [4, 4, 4, 128, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv2 = tf.get_variable('g_bconv1', [128], initializer=tf.constant_initializer(.1))
            h_conv2 = tf.nn.conv3d_transpose(h_relu1, w_conv2, output_shape=[self.batch_size, 8, 8, 8, 128], strides=[1, 2, 2, 2, 1], padding='SAME')+b_conv2
            h_norm2 = tf.contrib.layers.batch_norm(inputs = h_conv2, center=True, scale=True, is_training=is_training, scope="g_bn2")
            h_relu2 = tf.nn.relu(h_norm2)

            w_conv3 = tf.get_variable('g_wconv2', [4, 4, 4, 256, 128], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv3 = tf.get_variable('g_bconv2', [256], initializer=tf.constant_initializer(.1))
            h_conv3 = tf.nn.conv3d_transpose(h_relu2, w_conv3, output_shape=[self.batch_size, 16, 16, 16, 256], strides=[1, 2, 2, 2, 1], padding='SAME')+b_conv3
            h_norm3 = tf.contrib.layers.batch_norm(inputs = h_conv3, center=True, scale=True, is_training=is_training, scope="g_bn3")
            h_relu3 = tf.nn.relu(h_norm3)

            w_conv4 = tf.get_variable('g_wconv3', [4, 4, 4, 1, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_conv4 = tf.get_variable('g_bconv3', [1], initializer=tf.constant_initializer(.1))
            h_conv4 = tf.nn.conv3d_transpose(h_relu3, w_conv4, output_shape=[self.batch_size, 32, 32, 32, 1], strides=[1, 2, 2, 2, 1], padding='SAME')+b_conv4
            h_norm4 = tf.contrib.layers.batch_norm(inputs = h_conv4, center=True, scale=True, is_training=is_training, scope="g_bn4")

            #y = tf.sigmoid(h_conv4)

            return h_norm4

    def get_loss(self, x, y_hat_2d):

        x_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=x,logits=y_hat_2d))

        return (1.0/self.batch_size) * self.lambda_x * x_loss

    def generator(self, z, is_training=False, reuse=True):

       y_gene = self._decoder(z, is_training, reuse=reuse)
       return y_gene
