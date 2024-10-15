import os
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from data.augmentations import get_transform
from projects import PROJECT_DIR
import projects.dsine.config as config
import logging
import matplotlib.pyplot as plt
from projects import DATASET_DIR

logger = logging.getLogger('root')


class NormalDataset(Dataset):
    def __init__(self, args, dataset_name='hypersim', split='test', mode='test', epoch=0):
        self.args = args
        self.split = split
        self.mode = mode
        self.batch_size = args.batch_size
        assert mode in ['train', 'test']

        # data split
        #split_path = os.path.join(PROJECT_DIR, 'data', 'datasets', dataset_name, 'split', split+'.txt')
        if split == 'train':
            #split_path = '/home/amax/yuancai/datasets/study1/split/train_filenames.txt'
            split_path = os.path.join(DATASET_DIR,'split/train_filenames.txt')
        if split == 'test':
            #split_path = '/home/amax/yuancai/datasets/study1/split/test_filenames.txt'
            split_path = os.path.join(DATASET_DIR,'split/test_filenames.txt')
        #print('*'*10,split_path)
        assert os.path.exists(split_path)

        with open(split_path, 'r') as f:
            self.filenames = [i.strip() for i in f.readlines()]

        
        dataset_name='hypersim'
        # get_sample function
        if dataset_name == 'nyuv2':
            from data.datasets.nyuv2 import get_sample
        elif dataset_name == 'scannet':
            from data.datasets.scannet import get_sample
        elif dataset_name == 'ibims':
            from data.datasets.ibims import get_sample
        elif dataset_name == 'sintel':
            from data.datasets.sintel import get_sample
        elif dataset_name == 'vkitti':
            from data.datasets.vkitti import get_sample
        elif dataset_name == 'oasis':
            from data.datasets.oasis import get_sample
        elif dataset_name == 'hypersim':
            from data.datasets.hypersim.__init__ import get_sample    
        self.get_sample = get_sample

        # shuffle images
        if self.mode == 'train':
            logger.info('shuffling filenames with seed %s' % epoch)
            random.seed(epoch)
            random.shuffle(self.filenames)
            logger.info('accumulating gradients every %s batch' % args.accumulate_grad_batches)
            num_batches = len(self.filenames) // (args.batch_size * args.accumulate_grad_batches)
            num_imgs = num_batches * args.batch_size * args.accumulate_grad_batches
            self.filenames = self.filenames[:num_imgs]
            logger.info('%s imgs will be used / effective batch size: %s' % (num_imgs, args.batch_size * args.accumulate_grad_batches))

        # data preprocessing/augmentation
        self.transform = get_transform(args, dataset_name=dataset_name, mode=mode)
        
        '''
        # randomize intrinsics
        self.random_intrins = args.data_augmentation_intrins
        if self.mode == 'train' and self.random_intrins:
            logger.info('Random intrins: %s' % self.random_intrins)

            # aspect ratio will be randomly selected from below
            self.aspect_ratios = [
                (320, 960),
                (384, 800),
                (448, 672),
                (512, 608),
                (576, 544),
                (640, 480),
                (704, 448),
                (768, 416),
                (832, 384),
                (896, 352),
                (960, 320),
            ]
        else:
            logger.info('Random intrins: disabled')
        '''

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        #if self.mode == 'train' and self.random_intrins:
        if self.mode == 'train':
            random.seed(index // self.batch_size)
            self.aspect_ratios = [
                (320, 960),
                (384, 800),
                (448, 672),
                (512, 608),
                (576, 544),
                (640, 480),
                (704, 448),
                (768, 416),
                (832, 384),
                (896, 352),
                (960, 320),
            ]
            crop_H, crop_W = random.choice(self.aspect_ratios)
            info = {
                'crop_H': crop_H,
                'crop_W': crop_W,
            }
        else:
            info = {}
        sample1 = self.get_sample(
            args=self.args,
            sample_path=self.filenames[index], 
            info=info)
        # print(sample1,'**********')
        sample = self.transform(self.get_sample(
            args=self.args,
            sample_path=self.filenames[index], 
            info=info)
        )
        # print('**********',sample1.img)
        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.imshow((sample1.img+1)/2)
        # plt.title(sample1.img_name)     
        # plt.axis('off')
        # plt.subplot(2, 1, 2)
        # plt.imshow((sample1.normal+1)/2)
        # plt.axis('off')
        # plt.show()  

        #during training, we ensure that at least 1% of pixels have valid ground truth
        if self.mode == 'train':
            while sample['normal_mask'].sum() / sample['normal_mask'].numel() < 0.01:
                print('Replacing with another image... / num valid pixel: %s' % sample['normal_mask'].sum().item())
                new_index = random.randint(0, len(self.filenames)-1)
                sample = self.transform(self.get_sample(
                    args=self.args,
                    sample_path=self.filenames[new_index], 
                    info=info)
                )

        return sample            
    

class TrainLoader(object):
    def __init__(self, args, epoch=0):
        self.train_samples = NormalDataset(args, dataset_name=args.dataset_name_train, 
                                           split=args.train_split, mode='train', epoch=epoch)
    
  
        if args.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_samples, shuffle=False, drop_last=True)
        else:
            self.train_sampler = None

        self.data = DataLoader(self.train_samples, 
                               args.batch_size,
                               shuffle=False,
                               num_workers=args.num_workers,
                               pin_memory=True,
                               drop_last=True,
                               sampler=self.train_sampler)
        # for i, batch in enumerate(self.data):
        #     if i==0:
        #         plt.figure()
        #         plt.subplot(2,1,1)
        #         plt.imshow((batch['img'][0].permute(1,2,0)+1)/2)
        #         plt.title(batch['img_name'][0])     
        #         plt.axis('off')
        #         plt.subplot(2, 1, 2)
        #         plt.imshow((batch['normal'][0].permute(1,2,0)+1)/2)
        #         plt.axis('off')
        #         plt.show()  


class ValLoader(object):
    def __init__(self, args):
        self.val_samples = NormalDataset(args, dataset_name=args.dataset_name_val, 
                                         split=args.val_split, mode='test', epoch=None)
        self.data = DataLoader(self.val_samples, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


class TestLoader(object):
    def __init__(self, args):
        self.test_samples = NormalDataset(args, dataset_name=args.dataset_name_test, 
                                          split=args.test_split, mode='test', epoch=None)
        self.data = DataLoader(self.test_samples, 1, shuffle=False, num_workers=1, pin_memory=True)


if __name__ == '__main__':
    args = config.get_args()
    train_samples = NormalDataset(args,dataset_name='hypersim', 
                                        split='train', mode='train', epoch=0)
                                           

    train_sampler = None
    dataloader = DataLoader(args,train_samples, 
                               batch_size=4,
                               shuffle=False,
                               num_workers=32,
                               pin_memory=True,
                               drop_last=True,
                               sampler=train_sampler)

    for i, batch in enumerate(dataloader):
        if i == 0:  # 只显示第一批图像
            plt.figure(figsize=(10, 10))
            for idx, img in enumerate(batch):
                plt.subplot(1, len(batch), idx + 1)
                plt.imshow(img.permute(1, 2, 0))  # 调整通道顺序
                plt.title(f'Image {idx+1}')
                plt.axis('off')
            plt.show()
            break
