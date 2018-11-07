import torch
from pprint import pprint


class Option(object):
    """Configuration for training on Kaggle RSNA Pneumonia detection"""

    # root dir of training and validation set
    root_dir = '/media/storage/wu1114/RSNA/stage_2_train/'
    # test_dir = '/media/storage/wu1114/RSNA/stage_1_test/'
    test_dir = '/media/storage/wu1114/RSNA/stage_2_test/'
    result_dir = '/media/storage/wu1114/RSNA/stage_2_result/'

    batch_size = 1
    ngpu = 1
    imageSize = 64
    min_size = 600  # image resize
    max_size = 1000  # image resize
    nc = 3
    lr = 1e-3
    weight_decay = 0.0005
    betas = (0.5, 0.999)
    epoch = 100
    save_model = 1
    use_adam = False  # Use Adam optimizer
    use_drop = False  # use dropout in RoIHead
    hist_equalize = False
    shuffle = True
    which_pc = 1  # 0: train on civs linux, 1: train on p219, 2: macbook, 3: windows


    checkpoint_dir = './checkpoints/'
    # load_path = '/home/liming/.torch/models/vgg16-00b39a1b.pth'    # model path
    load_path = ''
    # load_path = True
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    pin_memory = True
    caffe_pretrain = True  # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    rpn_sigma = 3.
    roi_sigma = 1.
    test_num = 10000
    # test_num=100
    lr_decay = 0.5
    num_workers = 8
    test_num_workers = 8
    plot_every = 20  # vis every N iter
    env = 'RSNA-stage2'


    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')


    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in opt.__dict__.items() \
                if not k.startswith('_')}

opt = Option()