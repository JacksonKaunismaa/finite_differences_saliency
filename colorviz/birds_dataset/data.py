from torch.utils.data import Dataset,DataLoader
import os.path as osp
from tqdm import tqdm
import pandas as pd
from torchvision.io import read_image
import numpy as np
import torch
import glob
import warnings

from ..conv_color import config_objects

class ImageDataset(Dataset):
    def __init__(self, split, transform, cfg: config_objects.ImageDatasetCfg):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.transform = transform

        self.image_names = glob.glob(osp.join(cfg.data_dir, split, "*", "*.jpg"))
        self.idx_to_class_name = dict((i, osp.basename(fname)) for i, fname in enumerate(glob.glob(osp.join(cfg.data_dir, split, "*"))))
        self.class_name_to_idx = {c:i for i,c in self.idx_to_class_name.items()}
        self.num_classes = len(self.idx_to_class_name)
    
    def dataloader(self):
        return DataLoader(self, batch_size=self.cfg.batch_size, 
                          shuffle=True, num_workers=self.cfg.num_workers, pin_memory=True)

    def __len__(self):
        return len(self.image_names)
    
    def generate_one(self):
        idx = np.random.randint(len(self))
        return self.images[idx], self.labels[idx]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        im_name = self.image_names[idx]
        # bad practice, since we are replicating what transform.ToTensor does since that somehow doesn't work well with read_image
        image = read_image(im_name).float()/255.
        label = self.class_name_to_idx[osp.basename(osp.dirname(im_name))]
        if hasattr(self, "transform"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                image = self.transform(image)
        sample = {'image': image, 'label': label}
        return sample

    def implicit_normalization(self, inpt):
        # since TextureDatasetGenerator.generate_one returns np.uint8 (since it copies from a loaded image), transforms.ToTensor()
        # will implicitly do a divide by 255. Since we work in [0, 255] space for basically everything, this function is provided
        # for convenience and should be called just before we pass any tensor into the model. (This means that networks using this
        # dataset are working in [0, 1] space). The one exception to this is when
        # you have already called utils.tensorize on the image, which replicates the behaviour of transforms.ToTensor() (but with
        # better handling of adding dimensions/transposing) and thus will also implicitly do the divide by 255 operation (if the input
        # type is np.uint8). In summary, always call either utils.tensorize xor this function before passing into the model.
        return inpt/255.