import os
import time
import random
import pandas as pd
import numpy as np
from skimage import io
from PIL import Image
import pydicom
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import exposure

def read_image(path, dtype=np.float32, color=True):
    """
    Read an image from a given path, the image is in (C, H, W) format and the range of its value is between [0, 255]
    :param path: The path of an image file
    :param dtype: data type of an image, default is float32
    :param color: If 'True', the number of channels is three, in this case, RGB
        If 'False', this function returns a gray scale image
    :return:
    """
    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def vis_bbox(img_path, bbox_path):
    """
    Visualize one training image and its corresponding bbox
    :param img_path: full path of the training image
    :param bbox_path: full path of the bbox
    """
    image = io.imread(img_path)
    bbox = np.load(bbox_path)
    import matplotlib.patches as patches
    ax = plt.subplot(111)
    ax.imshow(image)
    for i in range(bbox.shape[0]):
        rect = patches.Rectangle((bbox[i][1], bbox[i][0]), bbox[i][3] - bbox[i][1], bbox[i][2] - bbox[i][0],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

def resize_bbox(bbox, in_size, out_size):
    """
    Resize bounding boxes according to image resize.
    :param bbox (numpy.ndarray): An array that has a shape (R, 4),
        'R' is the number of the bounding boxes.
        '4' indicates the position of the box, which represents
        (y_min, x_min, y_max, x_max)
    :param in_size (tuple): A tuple of length 2 which indicates
        the height and width of the image before resizing.
    :param out_size (tuple): A tuple of length 2 which indicates
        the height and width of the image after resizing.
    :return (numpy.ndarray): Rescaled bounding boxes according to
        the 'in_size'
    """
    bbox = bbox.copy()
    if len(bbox)==0:
        return bbox
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox


def flip_bbox(bbox, size, y_flip=False, x_flip=False):
    """
    Flip bounding boxes accordingly.
    :param bbox (numpy.ndarray): An array that has a shape (R, 4),
        'R' is the number of the bounding boxes.
        '4' indicates the position of the box, which represents
        (y_min, x_min, y_max, x_max)
    :param size (tuple): A tuple of length 2 which indicates
        the height and width of the image before resizing.
    :param y_flip (bool): Flip bounding box according to a vertical
        flip of an image.
    :param x_flip (bool): Flip bounding box according to a horizontal
        flip of an image.
    :return (numpy.ndarray): Bounding boxes flipped according to the given flips.
    """
    H, W = size
    bbox = bbox.copy()
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox

def flip_masks(masks, y_flip=False, x_flip=False):
    masks = masks.copy()
    for i in range(masks.shape[0]):
        if y_flip:
            masks[i] = np.flipud(masks[i]).copy()
        if x_flip:
            masks[i] = np.fliplr(masks[i]).copy()
    return masks

def random_flip(img, y_random=False, x_random=False,
                return_param=False, copy=False):
    """
    Randomly flip an image in vertical or horizontal direction.
    :param img (numpy.ndarray): An array that gets flipped.
        This is in CHW format.
    :param y_random (bool): Randomly flip in vertical direction.
    :param x_random (bool): Randomly flip in horizontal direction.
    :param return_param (bool): Returns information of flip.
    :param copy (bool): If False, a view of :obj:`img` will be returned.
    :return (numpy.ndarray or (numpy.ndarray, dict)):
        If :`return_param = False`,
        returns an array :obj:`out_img` that is the result of flipping.
        If :obj:`return_param = True`,
        returns a tuple whose elements are :obj:`out_img, param`.
        :obj:`param` is a dictionary of intermediate parameters whose
        contents are listed below with key, value-type and the description
        of the value.
        * **y_flip** (*bool*): Whether the image was flipped in the\
            vertical direction or not.
        * **x_flip** (*bool*): Whether the image was flipped in the\
            horizontal direction or not.
    """
    y_flip, x_flip = False, False
    if y_random:
        y_flip = random.choice([True, False])
    if x_random:
        x_flip = random.choice([True, False])

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    if copy:
        img = img.copy()

    if return_param:
        return img, {'y_flip': y_flip, 'x_flip': x_flip}
    else:
        return img

def read_dicom(src_root, dest_root):
    """
    Read dicom data and save them to png
    :param src_root:
    :param dest_root:
    :return:
    """
    print('Start converting dicom to png...')
    start = time.time()
    files = os.listdir(src_root)
    for file in files:
        if not os.path.exists(os.path.join(dest_root, file.split('.')[0])):
            os.mkdir(os.path.join(dest_root, file.split('.')[0]))
        path = os.path.join(src_root, file)
        d = pydicom.read_file(path)
        img = d.pixel_array
        io.imsave(os.path.join(dest_root, file.split('.')[0], 'image.png'), img)
    end = time.time()
    print('Total time usage: ', (end-start)/60., ' minutes')

def read_labels(src_root, dest_root):
    print('Start reading labels...')
    start = time.time()
    df = pd.read_csv(os.path.join(src_root, 'stage_2_train_labels.csv'))
    files = os.listdir(os.path.join(src_root, 'stage_2_train_images'))
    for file in files:
        patientId = file.split('.')[0]
        patient = df[df['patientId']==patientId]
        bbox = []
        if patient['Target'].iloc[0]==0:
            continue
        for i in range(len(patient)):
            x = patient['x'].iloc[i]
            y = patient['y'].iloc[i]
            width = patient['width'].iloc[i]
            height = patient['height'].iloc[i]
            bbox.append(np.array([y, x, y+height, x+width]).astype(np.float32))
        np.save(os.path.join(dest_root, patientId, 'bbox.npy'), np.array(bbox))
    end = time.time()
    print('Total time usage: ', (end-start)/60., ' minutes')

def to_hist_equal():
    """
    Read all images and save them to histogram equalized image.
    :return:
    """
    root = '/media/storage/wu1114/RSNA/stage_1_train/'
    files = os.listdir(root)
    for file in tqdm(files):
        print(file.split('.')[0])
        img = read_image(os.path.join(root, file.split('.')[0], 'image.png'), dtype=np.float32, color=True)
        hist_img = exposure.equalize_hist(img)
        hist_img = np.transpose(hist_img, (1, 2, 0))
        io.imsave(os.path.join(root, file.split('.')[0], 'hist_image.png'), hist_img)


if __name__ == '__main__':
    # Read dicom
    # src_root = '/media/storage/wu1114/RSNA/dataset/stage_2/stage_2_train_images/'
    # dest_root = '/media/storage/wu1114/RSNA/stage_2_train/'
    # read_dicom(src_root, dest_root)
    # src_root = '/home/PNW/wu1114/Documents/dataset/RSNA/stage_1_test_images'
    # dest_root = '/media/storage/wu1114/RSNA/stage_1_test'
    # read_dicom(src_root, dest_root)

    # Read labels
    src_root = '/media/storage/wu1114/RSNA/dataset/stage_2/'
    dest_root = '/media/storage/wu1114/RSNA/stage_2_train'
    read_labels(src_root, dest_root)

    # Convert to histtogram equalized image
    # to_hist_equal()