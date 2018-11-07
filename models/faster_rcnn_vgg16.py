import numpy as np
import torch as t
from torch import nn
from torch.autograd import Variable
from torchvision.models import vgg16
from models.region_proposal_network import RegionProposalNetwork
from models.faster_rcnn import FasterRCNN
from utils import array_tool as at
from utils.Config import opt
from roiAlign.roi_align.roi_align import RoIAlign


def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load('/home/PNW/wu1114/Documents/Faster-rcnn/checkpoints/vgg16_caffe.pth'))
    else:
        model = vgg16(not opt.load_path)

    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = True

    return nn.Sequential(*features), classifier


class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=1,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
        extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)
        self.fcn = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.mask = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        # self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)
        self.roi = RoIAlign(7, 7, transform_fpcoor=False)

    def forward(self, img_size, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        img_h, img_w = img_size
        # size of rois in the input images. (h, w)
        roi_size = np.concatenate((np.expand_dims(at.tonumpy(rois[:, 2] - rois[:, 0]), axis=1),
                                 (np.expand_dims(at.tonumpy(rois[:, 3] - rois[:, 1]), axis=1))), axis=1)
        feature_h, feature_w = x.shape[2], x.shape[3]
        roi_indices = at.totensor(roi_indices).int()
        rois = at.totensor(rois).float()
        rois[:, 0] = rois[:, 0] / img_h * feature_h
        rois[:, 2] = rois[:, 2] / img_h * feature_h
        rois[:, 1] = rois[:, 1] / img_w * feature_w
        rois[:, 3] = rois[:, 3] / img_w * feature_w
        # pool = self.roi(x, indices_and_rois)
        rois = at.tovariable(rois)
        roi_indices = at.tovariable(roi_indices)
        pool = self.roi(x, rois, roi_indices)  # (128, 512, 7, 7)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()