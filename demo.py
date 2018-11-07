import os
import torch as t
from utils.Config import opt
from models.faster_rcnn_vgg16 import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.data_utils import read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
from data.dataset import RSNADataset, inverse_normalize
from torch.utils import data as data_
from tqdm import tqdm


# Monkey-patch because I trained with a newer version.
# This can be removed once PyTorch 0.4.x is out.
# See https://discuss.pytorch.org/t/question-about-rebuild-tensor-v2/14560
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

#%%
faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()
trainer.load('./checkpoints/fasterrcnn_09031352_0')
opt.caffe_pretrain=True  # this model was trained from caffe-pretrained model
# Plot examples on training set
dataset = RSNADataset(opt.root_dir)
for i in range(0, len(dataset)):
    sample = dataset[i]
    img = sample['image']
    ori_img_ = inverse_normalize(at.tonumpy(img))

    # plot predicti bboxes
    _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
    pred_img = vis_bbox(ori_img_,
                           at.tonumpy(_bboxes[0]),
                           at.tonumpy(_labels[0]).reshape(-1),
                           at.tonumpy(_scores[0]))