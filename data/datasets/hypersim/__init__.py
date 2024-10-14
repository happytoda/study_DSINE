""" Get samples from NYUv2 (https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)
    NOTE: GT surface normals are from GeoNet (CVPR 2018) - https://github.com/xjqi/GeoNet
"""
import os
import cv2
import numpy as np
import torch
from data import Sample
import PIL.Image as Image
import torchvision.transforms as T

from projects import DATASET_DIR

# DATASET_PATH = os.path.join(DATASET_DIR, 'dsine_eval', 'nyuv2')
DATASET_PATH = '/home/amax/yuancai/datasets/study/'
# hypersim：/home/amax/yuancai/datasets/study/test/images/000005.png
# sample_path = "train/conditioning_images/020728.png train/images/020728.png"
resolution = 768


def get_sample(args, sample_path, info):
    # e.g. sample_path = "test/000000_img.png"
    condation_img_name, normal_img_name = sample_path.split()
    scene_name = sample_path.split('/')[0]
    img_name = sample_path.split('/')[2]
    img_ext = 'png'

    # img_path = '%s/%s' % (DATASET_PATH, sample_path)
    # normal_path = img_path.replace('_img'+img_ext, '_normal.png')
    # intrins_path = img_path.replace('_img'+img_ext, '_intrins.npy')
    img_path = os.path.join(DATASET_PATH, condation_img_name)
    normal_path = os.path.join(DATASET_PATH, normal_img_name)

    assert os.path.exists(img_path)
    assert os.path.exists(normal_path)
    # assert os.path.exists(intrins_path)

    # read image (H, W, 3)
    img = None
    if args.load_img:
        img = Image.open(img_path).convert('RGB')
        img_array = (np.array(img).astype(np.float32)) / 255.0
        source = torch.from_numpy(img_array)

    # read normal (H, W, 3)
    normal = normal_mask = None
    # transform = None
    transform = T.Compose([
        T.RandomResizedCrop(size=resolution, scale=(0.5, 2.0)),
    ])
    if args.load_normal:
        normal = Image.open(normal_path).convert('RGB')

        normal_array = ((np.array(normal).astype(np.float32)) / 255.0) * 2.0 - 1.0
        target = torch.from_numpy(normal_array)
    source = source.permute(2, 0, 1)
    target = target.permute(2, 0, 1)
    # print('!'*10,source.shape)
    # print('!'*10,target.shape)

    cat_data = torch.cat((source, target), 0)
    cat_data = transform(cat_data)

    source = cat_data[:3]
    target = cat_data[3:]

    target_norm = torch.norm(target, p=2, dim=0, keepdim=True)
    invalid_mask = (target_norm < 0.5) & (target_norm > 1.5)

    source = (source * 2.0) - 1.0
    target = (target * 2.0) - 1.0

    target1 = target.permute(1, 2, 0)
    target_array = target1.numpy()
    target_array = np.clip(target_array, -1, 1)
    target2 = torch.from_numpy(target_array)
    target3 = target2.permute(2, 0, 1)
    target = target3
    # target[invalid_mask.repeat(3, 1, 1)] = -1.

    source = source.permute(1, 2, 0)
    target = target.permute(1, 2, 0)

    if args.load_normal:
        normal_mask = torch.sum(target, axis=2, keepdims=True) > 0

    source = source.numpy()
    target = target.numpy()
    normal_mask = normal_mask.numpy()
    # print('*'*10,source.shape)
    # print('*'*10,target.shape)
    # print('*'*10,normal_mask.shape)
    # read intrins (3, 3)
    '''
    intrins = None
    if args.load_intrins:
        intrins = np.load(intrins_path)
    '''
    sample = Sample(
        img=source,
        normal=target,
        normal_mask=normal_mask,
        # intrins=intrins,
        intrins=target[:, :, 0],

        dataset_name='hypersim',
        scene_name=scene_name,
        img_name=img_name,
        info=info
    )

    return sample