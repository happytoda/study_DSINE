""" Get samples from NYUv2 (https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)
    NOTE: GT surface normals are from GeoNet (CVPR 2018) - https://github.com/xjqi/GeoNet
"""
import os
import cv2
import numpy as np

from data import Sample

from projects import DATASET_DIR
#DATASET_PATH = os.path.join(DATASET_DIR, 'dsine_eval', 'nyuv2')
DATASET_PATH = '/home/amax/yuancai/datasets/hypersim_normal'
#hypersim：/home/amax/yuancai/datasets/hypersim_normal/test/images/000005.png
#sample_path = "train/conditioning_images/020728.png train/images/020728.png"

def get_sample(args, sample_path, info):
    # e.g. sample_path = "test/000000_img.png" 
    condation_img_name,normal_img_name = sample_path.split() 
    scene_name = sample_path.split('/')[0]
    img_name= sample_path.split('/')[2]
    img_ext = 'png'

    #img_path = '%s/%s' % (DATASET_PATH, sample_path)
    #normal_path = img_path.replace('_img'+img_ext, '_normal.png')
    #intrins_path = img_path.replace('_img'+img_ext, '_intrins.npy')
    img_path = os.path.join(DATASET_PATH,condation_img_name)
    normal_path = os.path.join(DATASET_PATH,normal_img_name)
    assert os.path.exists(img_path)
    assert os.path.exists(normal_path)
    #assert os.path.exists(intrins_path)

    # read image (H, W, 3)
    img = None
    if args.load_img:
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

    # read normal (H, W, 3)
    normal = normal_mask = None
    if args.load_normal:
        normal = cv2.cvtColor(cv2.imread(normal_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        normal_mask = np.sum(normal, axis=2, keepdims=True) > 0
        normal = (normal.astype(np.float32) / 255.0) * 2.0 - 1.0

    # read intrins (3, 3)
    '''
    intrins = None
    if args.load_intrins:
        intrins = np.load(intrins_path)
    '''
    sample = Sample(
        img=img,
        normal=normal,
        normal_mask=normal_mask,
        #intrins=intrins,
        intrins = normal[:,:,0],

        dataset_name='hypersim',
        scene_name=scene_name,
        img_name=img_name,
        info=info
    )

    return sample