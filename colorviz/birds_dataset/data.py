from torch.utils.data import Dataset,DataLoader, default_collate
import os.path as osp
from tqdm import tqdm
import pandas as pd
from torchvision.io import read_image
import numpy as np
import torch
import glob
import warnings

from ..conv_color.config_objects import ImageDatasetCfg, ExperimentConfig

class ImageDataset(Dataset):
    def __init__(self, split, transform, cfg: ImageDatasetCfg, ddp=False):
        super().__init__()
        self.cfg = cfg
        self.ddp = ddp
        self.split = split
        self.transform = transform
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.GIF', '*.BMP']

        self.image_names = []
        for ext in image_extensions:
            self.image_names.extend(glob.glob(osp.join(cfg.data_dir, split, '*', ext)))

        class_names = glob.glob(osp.join(cfg.data_dir, split, "*"))
        class_names.sort()
        self.idx_to_class_name = {i: osp.basename(fname) for i, fname in enumerate(class_names)}
        self.class_name_to_idx = {c:i for i,c in self.idx_to_class_name.items()}
        self.num_classes = len(self.idx_to_class_name)
        print(split, "set size:", len(self))
    
    def dataloader(self):
        if self.ddp:
            sampler = torch.utils.data.DistributedSampler(self, shuffle=True)
        else:
            sampler = torch.utils.data.RandomSampler(self, replacement=False)

        # def ignore_nones_collate(batch):  # useful for if some images are rgb, some greyscale
        #     batch = list(filter(lambda x: x is not None, batch))
        #     return default_collate(batch)
        
        return DataLoader(self, batch_size=self.cfg.batch_size, sampler=sampler,
                          num_workers=self.cfg.num_workers, pin_memory=True)#, collate_fn=ignore_nones_collate)

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
        if np.random.randint(0,{'train': 50, 'valid': 2}[self.split]) == 0 and label in range(0,500,83):
            print("sampled image", im_name, "label", label, "in split", self.split)
        if hasattr(self, "transform"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    image = self.transform(image)
                except RuntimeError:
                    return None
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