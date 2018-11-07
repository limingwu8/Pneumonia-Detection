from __future__ import division
from tqdm import tqdm
from collections import defaultdict
import itertools
import numpy as np
from torch.autograd import Variable
from data.dataset import inverse_normalize
import six
from data.data_utils import resize_bbox
from skimage import transform
from utils import array_tool as at
from operator import is_not
from functools import partial


from models.model_utils.bbox_tools import bbox_iou


# helper function to calculate IoU
def iou(box1, box2):
    x11, y11, w1, h1 = box1
    x21, y21, w2, h2 = box2
    assert w1 * h1 > 0
    assert w2 * h2 > 0
    x12, y12 = x11 + w1, y11 + h1
    x22, y22 = x21 + w2, y21 + h2

    area1, area2 = w1 * h1, w2 * h2
    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2 - xi1) * (yi2 - yi1)
        union = area1 + area2 - intersect
        return intersect / union


def map_iou(boxes_true, boxes_pred, scores, thresholds=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    Mean average precision at differnet intersection over union (IoU) threshold

    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image.
                    bbox format: (x1, y1, w, h)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output:
        map: mean average precision of the image
    """

    # According to the introduction, images with no ground truth bboxes will not be
    # included in the map score unless there is a false positive detection (?)

    # return None if both are empty, don't count the image in final evaluation (?)
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None

    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]

    map_total = 0

    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1  # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1  # bt has no match, count as FN

        fp = len(boxes_pred) - len(matched_bt)  # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m

    return map_total / len(thresholds)


def eval_mAP(trainer, val_loader):
    tqdm.monitor_interval = 0
    mAP = []
    for ii, sample in tqdm(enumerate(val_loader)):
        if len(sample.keys()) == 5:
            img_id, img, bbox, scale, label = sample['img_id'], sample['image'], sample['bbox'], sample['scale'], \
                                                sample['label']
            img, bbox, label = img.cuda().float(), bbox.cuda(), label.cuda()
            img, bbox, label = Variable(img), Variable(bbox), Variable(label)

        else:
            img_id, img, scale = sample['img_id'], sample['image'], sample['scale']
            bbox = np.zeros((1, 0, 4))
            label = np.zeros((1, 0, 1))
            img = img.cuda().float()
            img = Variable(img)
        # if bbox is None:
        #     continue
        scale = at.scalar(scale)
        ori_img_ = inverse_normalize(at.tonumpy(img[0]))
        pred_boxes, pred_labels, pred_scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
        pred_boxes = pred_boxes[0]
        pred_labels = pred_labels[0]
        pred_scores = pred_scores[0]
        bbox = at.tonumpy(bbox[0])
        # Rescale back
        C, H, W = ori_img_.shape
        ori_img_ = transform.resize(ori_img_, (C, H * (1 / scale), W * (1 / scale)), mode='reflect')
        o_H, o_W = H * (1 / scale), W * (1 / scale)
        pred_boxes = resize_bbox(pred_boxes, (H, W), (o_H, o_W))
        bbox = resize_bbox(bbox, (H,W), (o_H, o_W))
        mAP.append(map_iou(bbox, pred_boxes, pred_scores))
        # if ii>=100:
        #     break

    mAP = np.array(mAP)
    mAP = mAP[mAP != np.array(None)].astype(np.float32)

    return np.mean(mAP)


if __name__ == '__main__':
    # simple test
    # box1 = [100, 100, 200, 200]
    # box2 = [100, 100, 300, 200]
    # print(iou(box1, box2))

    boxes_true = np.array([[100, 100, 200, 200], [500, 100, 100, 200], [100, 500, 100, 200]])
    boxes_pred = np.array([[100, 600, 200, 200], [120, 120, 200, 200], [500, 400, 100, 200]])
    scores = [0.8, 0.99, 0.6]

    map = map_iou(boxes_true, boxes_pred, scores)
    print(map)
