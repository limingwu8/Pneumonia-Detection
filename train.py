import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import ipdb
import matplotlib
matplotlib.use('Agg')

from tqdm import tqdm
import time
import numpy as np
from utils.Config import opt
from data.dataset import Dataset, DataLoader
from models.faster_rcnn_vgg16 import FasterRCNNVGG16
from models.faster_rcnn_resnet import FasterRCNNResNet50
from torch.autograd import Variable
import torch.nn as nn
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from data.dataset import inverse_normalize, get_train_loader, get_train_val_loader
from skimage import io, transform
from data.data_utils import resize_bbox
import pandas as pd
from collections import OrderedDict
from utils.eval_tool import eval_mAP
import resource

# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

def train_val():
    print('load data')
    train_loader, val_loader = get_train_val_loader(opt.root_dir, batch_size=opt.batch_size, val_ratio=0.1,
                                                    shuffle=opt.shuffle, num_workers=opt.num_workers,
                                                    pin_memory=opt.pin_memory)
    faster_rcnn = FasterRCNNVGG16()
    # faster_rcnn = FasterRCNNResNet50()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    # if opt.load_path:
    #     trainer.load(opt.load_path)
    #     print('load pretrained model from %s' % opt.load_path)

    # trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        tqdm.monitor_interval = 0
        for ii, sample in tqdm(enumerate(train_loader)):
            if len(sample.keys()) == 5:
                img_id, img, bbox, scale, label = sample['img_id'], sample['image'], sample['bbox'], sample['scale'], \
                                                    sample['label']
                img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
                img, bbox, label = Variable(img), Variable(bbox), Variable(label)

            else:
                img_id, img, bbox, scale, label = sample['img_id'], sample['image'], np.zeros((1, 0, 4)), \
                                                  sample['scale'], np.zeros((1, 0, 1))
                img = img.cuda().float()
                img = Variable(img)

            if bbox.size == 0:
                continue

            scale = at.scalar(scale)
            trainer.train_step(img_id, img, bbox, label, scale)
            if (ii + 1) % opt.plot_every == 0:
                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot ground truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     img_id[0],
                                     at.tonumpy(bbox[0]),
                                     at.tonumpy(label[0]))

                trainer.vis.img('gt_img', gt_img)

                # plot predicted bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       img_id[0],
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))

                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())

        mAP = eval_mAP(trainer, val_loader)
        trainer.vis.plot('val_mAP', mAP)
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(mAP),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)
        if mAP > best_map:
            best_map = mAP
            best_path = trainer.save(best_map=best_map)
        if epoch==opt.epoch-1:
            best_path = trainer.save()

        if (epoch+1) % 10 == 0:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

def train(**kwargs):
    # opt._parse(kwargs)

    print('load data')
    dataloader = get_train_loader(opt.root_dir, batch_size=opt.batch_size,
                                  shuffle=opt.shuffle, num_workers=opt.num_workers,
                                  pin_memory=opt.pin_memory)
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    # if opt.load_path:
    #     trainer.load(opt.load_path)
    #     print('load pretrained model from %s' % opt.load_path)

    # trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, sample in tqdm(enumerate(dataloader)):
            if len(sample.keys()) == 5:
                img_id, img, bbox_, scale, label_ = sample['img_id'], sample['image'], sample['bbox'], sample['scale'], \
                                                    sample['label']
                img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
                img, bbox, label = Variable(img), Variable(bbox), Variable(label)

            else:
                img_id, img, bbox, scale, label = sample['img_id'], sample['image'], np.zeros((1, 0, 4)), \
                                                  sample['scale'], np.zeros((1, 0, 1))
                img = img.cuda().float()
                img = Variable(img)

            # if label.size == 0:
            #     continue

            scale = at.scalar(scale)
            trainer.train_step(img_id, img, bbox, label, scale)
            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot ground truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)


                # plot predicted bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())

        if epoch % 10 == 0:
            best_path = trainer.save(best_map=best_map)


if __name__ == '__main__':
    # import fire

    # fire.Fire()
    start = time.time()
    # train()
    train_val()
    end = time.time()
    print('total time: ', (end-start)/3600., ' hours')
