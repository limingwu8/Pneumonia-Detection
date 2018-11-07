import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
import time
import numpy as np
from utils.Config import opt
from models.faster_rcnn_vgg16 import FasterRCNNVGG16
from models.faster_rcnn_resnet import FasterRCNNResNet50
from torch.autograd import Variable
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox, rescale_back, save_gt_pred, save_pred_fig
from data.dataset import inverse_normalize, get_train_loader, get_test_loader, get_train_val_loader
from skimage import io, transform
from data.data_utils import resize_bbox
import pandas as pd
from collections import OrderedDict
from utils.eval_tool import eval_mAP
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


def pred_test():
    faster_rcnn = FasterRCNNVGG16()
    # faster_rcnn = FasterRCNNResNet50()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    # trainer.load('./checkpoints/RSNA_skip_09111650_0.19862726')  # 0.062
    # trainer.load('./checkpoints/fasterrcnn_09102119_0.2340059')  # 0
    # trainer.load('./checkpoints/RSNA_skip_09100834_0.16078612')  # 0
    # trainer.load('./checkpoints/RSNA_skip_10252107_0.22194205')  # 0.041
    trainer.load('./checkpoints/RSNA_no_skip_09131705_0.33119902')  # 0.089
    # trainer.load('./checkpoints/RSNA_no_skip_09162308_0.21759672')  # 0.015
    # trainer.load('./checkpoints/RSNA_skip_10011111')  # 0
    # trainer.load('./checkpoints/RSNA_skip_10270402')

    opt.caffe_pretrain = True  # this model was trained from caffe-pretrained model
    # Plot examples on training set
    print('load data')
    testloader = get_test_loader(opt.test_dir, batch_size=opt.batch_size,
                                  shuffle=opt.shuffle, num_workers=opt.num_workers,
                                  pin_memory=opt.pin_memory)
    patientId = []
    PredictionString = []
    for ii, sample in tqdm(enumerate(testloader)):
        img_id, img, bbox, scale, label = sample['img_id'], sample['image'], np.zeros((1, 0, 4)), \
                                          sample['scale'], np.zeros((1, 0, 1))
        scale = at.scalar(scale)

        img = at.tonumpy(img)[0]

        # plot predicti bboxes
        img = inverse_normalize(at.tonumpy(img[0]))
        pred_boxes, pred_labels, pred_scores = trainer.faster_rcnn.predict([img], visualize=True)
        pred_boxes = at.tonumpy(pred_boxes[0])
        pred_labels = at.tonumpy(pred_labels[0]).reshape(-1)
        pred_scores = at.tonumpy(pred_scores[0])

        # Rescale back
        img, bbox, pred_boxes = rescale_back(img, at.tonumpy(bbox[0]), pred_boxes, scale)

        save_path = os.path.join(opt.result_dir, 'pred_on_test_skip', img_id[0] + '.png')
        save_pred_fig(img, pred_boxes, pred_scores, img_id, save_path)

        # Save Info
        patientId.append(img_id[0])
        tmp = []
        for i in range(pred_boxes.shape[0]):
            y0, x0, y1, x1 = pred_boxes[0][0], pred_boxes[0][1], pred_boxes[0][2], pred_boxes[0][3]
            h = y1-y0
            w = x1-x0
            tmp.append([str(round(pred_scores[i],2)), ' ', str(int(x0)), ' ', str(int(y0)), ' ', str(int(w)), ' ', str(int(h)), ' '])
        pre_str = ''.join([item for sublist in tmp for item in sublist])
        PredictionString.append(pre_str[:-1])

    df = pd.DataFrame(OrderedDict((('patientId', pd.Series(patientId)), ('PredictionString', pd.Series(PredictionString)))))
    df.to_csv(os.path.join(opt.result_dir, 'pred_on_test_skip.csv'), index=False)

def pred_train():
    faster_rcnn = FasterRCNNVGG16()
    # faster_rcnn = FasterRCNNResNet50()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    trainer.load('./checkpoints/RSNA_skip_11061805_0.27202734')

    opt.caffe_pretrain = True  # this model was trained from caffe-pretrained model
    # Plot examples on training set
    print('load data')
    train_loader, val_loader = get_train_val_loader(opt.root_dir, batch_size=opt.batch_size, val_ratio=0.1,
                                                    shuffle=opt.shuffle, num_workers=opt.num_workers,
                                                    pin_memory=opt.pin_memory)
    patientId = []
    PredictionString = []
    for ii, sample in tqdm(enumerate(val_loader)):
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

        scale = at.scalar(scale)

        img = inverse_normalize(at.tonumpy(img[0]))

        pred_boxes, pred_labels, pred_scores = trainer.faster_rcnn.predict([img], visualize=True)
        pred_boxes = at.tonumpy(pred_boxes[0])
        pred_labels = at.tonumpy(pred_labels[0]).reshape(-1)
        pred_scores = at.tonumpy(pred_scores[0])

        # Rescale back
        img, bbox, pred_boxes = rescale_back(img, at.tonumpy(bbox[0]), pred_boxes, scale)

        # Save predicted images
        save_path = os.path.join(opt.result_dir, 'pred_on_val_skip', img_id[0] + '.png')
        save_gt_pred(img, bbox, pred_boxes, pred_scores, img_id, save_path)

        # Save Info
        patientId.append(img_id[0])
        tmp = []
        for i in range(pred_boxes.shape[0]):
            y0, x0, y1, x1 = pred_boxes[0][0], pred_boxes[0][1], pred_boxes[0][2], pred_boxes[0][3]
            h = y1-y0
            w = x1-x0
            tmp.append([str(pred_scores[i]), ' ', str(x0), ' ', str(y0), ' ', str(w), ' ', str(h), ' '])
        pre_str = ''.join([item for sublist in tmp for item in sublist])
        PredictionString.append(pre_str[:-1])

    df = pd.DataFrame(OrderedDict((('patientId', pd.Series(patientId)), ('PredictionString', pd.Series(PredictionString)))))
    df.to_csv(os.path.join(opt.result_dir, 'pred_on_val_skip.csv'), index=False)


if __name__ == '__main__':
    start = time.time()
    pred_train()  # Predict the result on training set
    # pred_test()  # Predict the result on testing set
    end = time.time()
    print('total time: ', (end-start)/3600., ' hours')
