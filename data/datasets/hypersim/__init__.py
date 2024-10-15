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
import matplotlib.pyplot as plt

from projects import DATASET_DIR
#DATASET_PATH = os.path.join(DATASET_DIR, 'dsine_eval', 'nyuv2')
DATASET_PATH = '/home/amax/yuancai/datasets/study1/'
#hypersim：/home/amax/yuancai/datasets/study/test/images/000005.png
#sample_path = "train/conditioning_images/020728.png train/images/020728.png"
resolution = 768

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
        img = Image.open(img_path).convert('RGB')
        img_array = (np.array(img).astype(np.float32))/ 255.0
        source = torch.from_numpy(img_array)
   
    # read normal (H, W, 3)
    normal = normal_mask = None
    #transform = None
    transform = T.Compose([
                        T.RandomResizedCrop(size=resolution, scale=(0.5, 2.0)),
                         ])
    if args.load_normal:
        normal = Image.open(normal_path).convert('RGB')
        
        normal_array = ((np.array(normal).astype(np.float32))/ 255.0)*2.0-1.0
        target = torch.from_numpy(normal_array)
    source = source.permute(2,0,1)
    target = target.permute(2,0,1)
    #print('!'*10,source.shape)
    #print('!'*10,target.shape)

    cat_data = torch.cat((source, target),0)
    cat_data = transform(cat_data)
    
    source01=source
    target01=target
    
            
    source = cat_data[:3]
    target = cat_data[3:]

    target_norm = torch.norm(target, p=2, dim=0, keepdim=True)
    invalid_mask = (target_norm < 0.5) | (target_norm > 1.5)
    #print('***************',invalid_mask.shape) (1,768,768)
    plt.figure()
    plt.imshow((invalid_mask.permute(1,2,0)+1)/2)
    plt.axis('off')
    plt.show()

    # source = (source * 2.0) - 1.0
    # target = (target * 2.0) - 1.0
    
    target1 = target.permute(1,2,0)
    target_array = target1.numpy()
    target_array = np.clip(target_array,-1,1)
    target2 = torch.from_numpy(target_array)
    target3=target2.permute(2,0,1)
    target = target3 
    #target[invalid_mask.repeat(3, 1, 1)] = -1.

    source = source.permute(1, 2, 0)
    target = target.permute(1, 2, 0)
    
    source02=source
    target02=target
  

    if args.load_normal:
        normal_mask = torch.sum(target, axis=2, keepdims=True) > 0
   
    source = source.numpy()
    target = target.numpy()
    normal_mask = normal_mask.numpy()
    #print('*'*10,source.shape)
    #print('*'*10,target.shape)
    #(768, 768, 3)
    print('*'*10,normal_mask.shape)
    # read intrins (3, 3)
    orig_H, orig_W,_ = target.shape
    intrins = intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W,).unsqueeze(0)
    lrtb = get_padding(orig_H, orig_W)
    intrins[:, 0, 2] += lrtb[0]
    intrins[:, 1, 2] += lrtb[2]
    intrins = intrins.numpy()
    intrins = intrins.squeeze()
    #print('******',intrins.shape)
    '''
    intrins = None
    if args.load_intrins:
        intrins = np.load(intrins_path)
    '''
    # plt.figure()
    # plt.subplot(2,3,1)
    # plt.imshow((source01.permute(1,2,0)+1)/2)
    # plt.axis('off')
    # plt.title(img_name)
    # plt.subplot(2,3,2)
    # plt.imshow((source02+1)/2)
    # plt.axis('off')
    # plt.subplot(2,3,3)
    # plt.imshow((source+1)/2)
    # plt.axis('off')
    # plt.subplot(2, 3, 4)
    # plt.imshow((target01.permute(1,2,0)+1)/2)
    # plt.axis('off')
    # plt.subplot(2, 3, 5)
    # plt.imshow((target02+1)/2)
    # plt.axis('off')
    # plt.subplot(2, 3, 6)
    # plt.imshow((target+1)/2)
    # plt.axis('off')
    # plt.show()

    sample = Sample(
        img=source,
        normal=target,
        normal_mask=normal_mask,
        intrins = intrins,
        dataset_name='hypersim',
        scene_name=scene_name,
        img_name=img_name,
        info=info
    )

    return sample

def intrins_from_fov(new_fov, H, W, dtype=torch.float32):
    """ define intrins based on field-of-view
        principal point is assumed to be at the center

        NOTE: new_fov should be in degrees
        NOTE: top-left is (0,0)
    """
    new_fx = new_fy = (max(H, W) / 2.0) / np.tan(np.deg2rad(new_fov / 2.0))
    new_cx = (W / 2.0) - 0.5
    new_cy = (H / 2.0) - 0.5

    new_intrins = torch.tensor([
        [new_fx,    0,          new_cx  ],
        [0,         new_fy,     new_cy  ],
        [0,         0,          1       ]
    ], dtype=dtype)

    return new_intrins

def get_padding(orig_H, orig_W):
    """ returns how the input of shape (orig_H, orig_W) should be padded
        this ensures that both H and W are divisible by 32
    """
    if orig_W % 32 == 0:
        l = 0
        r = 0
    else:
        new_W = 32 * ((orig_W // 32) + 1)
        l = (new_W - orig_W) // 2
        r = (new_W - orig_W) - l

    if orig_H % 32 == 0:
        t = 0
        b = 0
    else:
        new_H = 32 * ((orig_H // 32) + 1)
        t = (new_H - orig_H) // 2
        b = (new_H - orig_H) - t
    return l, r, t, b
