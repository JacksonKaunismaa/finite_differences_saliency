from colorviz.birds_dataset.data import ImageDataset
from colorviz.conv_color.config_objects import ImageDatasetCfg
# import keras
# import tensorflow as tf
import pickle
from torchvision.models import EfficientNet_B0_Weights


from colorviz.conv_color import visualizations 
from colorviz.birds_dataset.data import ImageDataset
from colorviz.conv_color.config_objects import ImageDatasetCfg

# tr_transform = transforms.Compose([transforms.Resize(256, antialias=True),
#                                 transforms.RandomCrop(224),
#                                 #transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ???
#                                                         std=[0.229, 0.224, 0.225]),
#                                 transforms.RandomHorizontalFlip(),
#                                 transforms.RandomRotation(8.),
#                                 transforms.ColorJitter(contrast=0.1)
#                                                         ])


# load_model_state_dict(net, optim=opt, sched=combined_sched)


va_transform = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
transform_map = dict(train=va_transform,
                    valid=va_transform,
                    test=va_transform)


data_cfg = ImageDatasetCfg(batch_size=512,
                            num_workers=4,
                            data_dir="/scratch/ssd004/scratch/jackk/birds_data",
                            device="cpu")


dsets = {split: ImageDataset(split, transform_map[split], data_cfg, ddp=False) for split in ["train", "valid", "test"]}
default_scales = [3,5,7,9,13,15]
pca_dirs = visualizations.find_pca_directions(dsets['train'], 8192, default_scales, 1)

with open("big_sample_pca_dirs.pkl", "wb") as p:
    pickle.dump(pca_dirs, p)