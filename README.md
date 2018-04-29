This project is a part of my bachelor degree paper,  and it is an implementation adapted from the network architecture in  [Learning a Predictable and Generative Vector Representation for Objects](https://arxiv.org/abs/1603.08637).

#### Usage

please download the pretrained [Resnet50v2](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained) model of slim, and put it into resnet_v2_50 folder.

The data used in training is downloaded from [Github:lsm](https://github.com/akar43/lsm), download it and put it into data folder. data/split.json includes the information about train and test dataset. cls_info includes the information about the classid and classname.

##### train model

```
python3 train.py -c [class id] -g [gpu id] -i [batch_size]
```

##### train model on multi gpus

```
python3 train.py -c [class id] -g [gpu id] -i [img per gpu]
#example: train the class bench with batch size 60 on 4 gpus
python3 train.py -c 02828884 -g 0123 -i 15
```

#### Network Architecture

![net](examples/net.png)

#### Results

![airplane1](examples/airplane1.png)
![airplane2](examples/airplane2.png)
![airplane3](examples/airplane3.png)
![airplane4](examples/airplane4.png)
![airplane5](examples/airplane5.png)
![airplane6](examples/airplane6.png)
![airplane7](examples/airplane7.png)
![airplane8](examples/airplane8.png)
![airplane9](examples/airplane9.png)
![airplane10](examples/airplane10.png)
![bench1](examples/bench1.png)
![bench2](examples/bench2.png)
![bench3](examples/bench3.png)
![bench4](examples/bench4.png)
![bench5](examples/bench5.png)
![chair1](examples/chair1.png)
![chair2](examples/chair2.png)
![chair3](examples/chair3.png)
![chair4](examples/chair4.png)
![chair5](examples/chair5.png)
![chair6](examples/chair6.png)
![chair7](examples/chair7.png)
![soft1](examples/soft1.png)
![soft2](examples/soft2.png)
![soft3](examples/soft3.png)
![soft4](examples/soft4.png)
![soft5](examples/soft5.png)
![soft6](examples/soft6.png)
![tabel1](examples/tabel1.png)
![tabel2](examples/tabel2.png)
![tabel3](examples/tabel3.png)
![tabel4](examples/tabel4.png)
![tabel5](examples/tabel5.png)
![tabel6](examples/tabel6.png)
![tabel7](examples/tabel7.png)
![tabel8](examples/tabel8.png)
![tabel9](examples/tabel9.png)
![tabel10](examples/tabel10.png)

##### ......

![1568501138](examples/1568501138.jpg)