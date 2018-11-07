import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from utils.Config import opt
from skimage import exposure
import matplotlib.pylab as plt
from utils import array_tool as at
from sklearn.model_selection import train_test_split
from data.data_utils import read_image, resize_bbox, flip_bbox, random_flip, flip_masks
from utils.vis_tool import apply_mask_bbox
import matplotlib.patches as patches

DSB_BBOX_LABEL_NAMES = ('p')  # Pneumonia

def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :].clip(min=0, max=255)
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255

"""Transforms:
Data augmentation
"""
class Transform(object):
    def __init__(self, min_size=600, max_size=1000, train=True):
        self.min_size = min_size
        self.max_size = max_size
        self.train = train

    def __call__(self, in_data):
        if len(in_data.keys())!=2:
            img_id, img, bbox, label = in_data['img_id'], in_data['image'], in_data['bbox'], in_data['label']
            _, H, W = img.shape
            img = preprocess(img, self.min_size, self.max_size, self.train)
            _, o_H, o_W = img.shape
            scale = o_H/H
            # horizontally flip
            # img, params = random_flip(img, x_random=True, y_random=True, return_param=True)

            bbox = resize_bbox(bbox, (H, W), (o_H, o_W))
            img, params = random_flip(img, x_random=True, y_random=False, return_param=True)
            bbox = flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'], y_flip=params['y_flip'])

            label = label if label is None else label.copy()
            return {'img_id': img_id, 'image': img.copy(), 'bbox': bbox, 'label': label, 'scale': scale}

        else:
            img_id, img = in_data['img_id'], in_data['image']
            _, H, W = img.shape
            img = preprocess(img, self.min_size, self.max_size, self.train)
            _, o_H, o_W = img.shape
            scale = o_H/H
            # horizontally flip
            # img, params = random_flip(img, x_random=True, y_random=True, return_param=True)
            return {'img_id': img_id, 'image': img.copy(), 'scale': scale}


def preprocess(img, min_size=600, max_size=1000, train=True):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze

    if opt.hist_equalize:
        hist_img = exposure.equalize_hist(img)
        hist_img = transform.resize(hist_img, (C, H * scale, W * scale), mode='reflect')
        hist_img = normalize(hist_img)
        return hist_img

    img = img / 255.
    img = transform.resize(img, (C, H * scale, W * scale), mode='reflect')
    # both the longer and shorter should be less than
    # max_size and min_size

    img = normalize(img)

    return img

def pytorch_normalze(img):
    """
    https://discuss.pytorch.org/t/how-to-preprocess-input-for-pre-trained-networks/683
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()


def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img

class RSNADataset(Dataset):
    def __init__(self, root_dir, img_id, transform=True, train=True):
        """
        Args:
        :param root_dir (string): Directory with all the images
        :param img_id (list): lists of image id
        :param train: if equals true, then read training set, so the output is image, mask and imgId
                      if equals false, then read testing set, so the output is image and imgId
        :param transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.img_id = img_id
        self.transform = transform
        self.tsf = Transform(opt.min_size, opt.max_size, train)

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_id[idx], 'image.png')
        bbox_path = os.path.join(self.root_dir, self.img_id[idx], 'bbox.npy')
        image = read_image(img_path, np.float32, True)
        if os.path.exists(bbox_path):
            bbox = np.load(bbox_path)
            label = np.zeros(len(bbox)).astype(np.int32)
            sample = {'img_id': self.img_id[idx], 'image':image.copy(), 'bbox':bbox, 'label': label}
        else:
            sample = {'img_id': self.img_id[idx], 'image':image.copy()}

        if self.transform:
            sample = self.tsf(sample)

        return sample


class RSNADatasetTest(Dataset):
    def __init__(self, root_dir, transform=True, train=False):
        """
        Args:
        :param root_dir (string): Directory with all the images
        :param img_id (list): lists of image id
        :param train: if equals true, then read training set, so the output is image, mask and imgId
                      if equals false, then read testing set, so the output is image and imgId
        :param transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.img_id = os.listdir(root_dir)
        self.transform = transform
        self.tsf = Transform(opt.min_size, opt.max_size, train)

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_id[idx], 'image.png')
        image = read_image(img_path, np.float32, True)

        sample = {'img_id': self.img_id[idx], 'image': image.copy()}

        if self.transform:
            sample = self.tsf(sample)

        return sample



def get_train_loader(root_dir, batch_size=16, shuffle=False, num_workers=4, pin_memory=False):

    """Utility function for loading and returning training and validation Dataloader
    :param root_dir: the root directory of data set
    :param batch_size: batch size of training and validation set
    :param split: if split data set to training set and validation set
    :param shuffle: if shuffle the image in training and validation set
    :param num_workers: number of workers loading the data, when using CUDA, set to 1
    :param val_ratio: ratio of validation set size
    :param pin_memory: store data in CPU pin buffer rather than memory. when using CUDA, set to True
    :return:
        if split the data set then returns:
        - train_loader: Dataloader for training
        - valid_loader: Dataloader for validation
        else returns:
        - dataloader: Dataloader of all the data set
    """
    img_ids = os.listdir(root_dir)
    img_ids.sort()
    transformed_dataset = RSNADataset(root_dir=root_dir, img_id=img_ids, transform=True, train=True)
    dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return dataloader

def get_train_val_loader(root_dir, batch_size=16, val_ratio=0.2, shuffle=False, num_workers=4, pin_memory=False):

    """Utility function for loading and returning training and validation Dataloader
    :param root_dir: the root directory of data set
    :param batch_size: batch size of training and validation set
    :param split: if split data set to training set and validation set
    :param shuffle: if shuffle the image in training and validation set
    :param num_workers: number of workers loading the data, when using CUDA, set to 1
    :param val_ratio: ratio of validation set size
    :param pin_memory: store data in CPU pin buffer rather than memory. when using CUDA, set to True
    :return:
        if split the data set then returns:
        - train_loader: Dataloader for training
        - valid_loader: Dataloader for validation
        else returns:
        - dataloader: Dataloader of all the data set
    """
    img_ids = os.listdir(root_dir)
    img_ids.sort()
    train_id, val_id = train_test_split(img_ids, test_size=val_ratio, random_state=55, shuffle=shuffle)
    train_dataset = RSNADataset(root_dir=root_dir, img_id=train_id, transform=True, train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_dataset = RSNADataset(root_dir=root_dir, img_id=val_id, transform=True, train=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader

def get_test_loader(test_dir, batch_size=16, shuffle=False, num_workers=4, pin_memory=False):

    """Utility function for loading and returning training and validation Dataloader
    :param root_dir: the root directory of data set
    :param batch_size: batch size of training and validation set
    :param shuffle: if shuffle the image in training and validation set
    :param num_workers: number of workers loading the data, when using CUDA, set to 1
    :param pin_memory: store data in CPU pin buffer rather than memory. when using CUDA, set to True
    :return:
        - testloader: Dataloader of all the test set
    """
    transformed_dataset = RSNADatasetTest(root_dir=test_dir)
    testloader = DataLoader(transformed_dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return testloader

def show_batch_train(sample_batched):
    """
    Visualize one training image and its corresponding bbox
    """
    if len(sample_batched.keys())==5:
        # if sample_batched['img_id']=='8d978e76-14b9-4d9d-9ba6-aadd3b8177ce':
        #     print('stop')
        img_id, image, bbox = sample_batched['img_id'], sample_batched['image'], sample_batched['bbox']
        orig_img = at.tonumpy(image)
        orig_img = inverse_normalize(orig_img)
        bbox = bbox[0, :]

        ax = plt.subplot(111)
        ax.imshow(np.transpose(np.squeeze(orig_img / 255.), (1, 2, 0)))
        ax.set_title(img_id[0])
        for i in range(bbox.shape[0]):
            y1, x1, y2, x2 = int(bbox[i][0]), int(bbox[i][1]), int(bbox[i][2]), int(bbox[i][3])
            h = y2 - y1
            w = x2 - x1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()

def show_batch_test(sample_batch):
    img_id, image = sample_batch['img_id'], sample_batch['image']
    image = inverse_normalize(at.tonumpy(image[0]))
    plt.figure()
    plt.imshow(np.transpose(at.tonumpy(image/255), (1, 2, 0)))
    plt.show()


if __name__ == '__main__':

    # dataset = RSNADataset(root_dir=opt.root_dir, transform=True)
    # sample = dataset[13]
    # print(sample.keys())

    # Load training set
    # trainloader = get_train_loader(opt.root_dir, batch_size=opt.batch_size, shuffle=opt.shuffle,
    #                                num_workers=opt.num_workers, pin_memory=opt.pin_memory)
    #
    # for i_batch, sample in tqdm(enumerate(trainloader)):
    #     B,C,H,W = sample['image'].shape
    #     if (H,W)!=(600,600):
    #         print(sample['img_id'])
    #     show_batch_train(sample)


    # Load testing set
    # testloader = get_test_loader(opt.test_dir, batch_size=opt.batch_size, shuffle=opt.shuffle,
    #                              num_workers=opt.num_workers, pin_memory=opt.pin_memory)
    # for i_batch, sample in enumerate(testloader):
    #     print('i_batch: ', i_batch, 'len(sample)', len(sample.keys()))
    #     show_batch_test(sample)

    # Load training & validation set
    train_loader, val_loader = get_train_val_loader(opt.root_dir, batch_size=opt.batch_size, val_ratio=0.1,
                                                    shuffle=True, num_workers=opt.num_workers,
                                                    pin_memory=opt.pin_memory)
    for i_batch, sample in enumerate(train_loader):
        show_batch_train(sample)


    # Test train & validation set on densenet
    # img_ids = os.listdir(opt.root_dir)
    # dataset = RSNADataset_densenet(root_dir=opt.root_dir, img_id=img_ids, transform=True)
    # sample = dataset[13]
    # print(sample.keys())


    # train_loader, val_loader = get_train_val_loader_densenet(opt.root_dir, batch_size=128, val_ratio=0.1,
    #                                                     shuffle=False, num_workers=opt.num_workers,
    #                                                     pin_memory=opt.pin_memory)
    # non_zeros = 0  # 4916 + 743 = 5659
    # zeros = 0  # 15692 + 4505 = 20197
    # for i, sample in tqdm(enumerate(val_loader)):
    #     non_zeros += np.count_nonzero(at.tonumpy(sample['label']))
    #     zeros += (128-np.count_nonzero(at.tonumpy(sample['label'])))
    #     # print(sample['img_id'], ', ', at.tonumpy(sample['label']))
    # print("non_zeros: ", non_zeros)
    # print("zeros: ", zeros)
