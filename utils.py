import os
import numpy as np
import scipy.io as scio
import cv2
import random

class DataSet():

    def __init__(self, **kwargs):

        self.__dict__.update(kwargs)

    def __call__(self):
        voxels = []
        render_imgs = []
        for fn in self.data_list:
            voxel_path = os.path.join(self.DATA_DIR,"voxels","modelVoxels"+str(32),self.train_cls,fn+".mat")
            render_path = os.path.join(self.DATA_DIR,"renders",self.train_cls,fn)
            voxels.append(np.expand_dims(scio.loadmat(voxel_path)['Volume'],3))
            imgs = []
            for i in range(20):
                img = cv2.imread(os.path.join(render_path,"render_{}.png".format(i)))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if isinstance(self.img_fixed_size, tuple):
                    img = cv2.resize(img, self.img_fixed_size)
                imgs.append(img)
            render_imgs.append(imgs)

        self.voxels = np.asarray(voxels)
        self.render_imgs = np.asarray(render_imgs)

    def _feed_dict(self, batch_size, x, imgs, is_training=True):
        batch_list = np.random.choice(len(self.data_list), batch_size)
        _dict = {
            #is_training: is_training,
        }
        for i in range(batch_size):
            _dict[x[i]]=self.voxels[batch_list[i]]
            ri = random.randint(0, 19)
            _dict[imgs[i]]=self.render_imgs[batch_list[i]][ri]
        return _dict

    def _feed_all_gpu(self,models,IMG_PER_GPU):
        _dict = {
            #is_training: is_training,
        }
        for i in range(len(models)):
            imgs, x, _, _, _, _, _, _ = models[i]
            batch_list = np.random.choice(len(self.data_list), IMG_PER_GPU)
            for j in range(IMG_PER_GPU):
                _dict[x[j]]=self.voxels[batch_list[j]]
                ri = random.randint(0, 19)
                _dict[imgs[j]]=self.render_imgs[batch_list[j]][ri]
        return _dict
