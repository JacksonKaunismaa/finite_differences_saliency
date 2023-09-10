from colorviz.birds_dataset.data import ImageDataset
from colorviz.conv_color.config_objects import ImageDatasetCfg
# import keras
# import tensorflow as tf
import pickle
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


from colorviz.conv_color import visualizations, hooks
from colorviz.birds_dataset.data import ImageDataset
from colorviz.conv_color.config_objects import ImageDatasetCfg, ExperimentConfig
from colorviz.birds_dataset import network

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
exp_cfg = ExperimentConfig(epochs=30,
                            lr_max=4e-3,
                            step_size=7,
                            weight_decay=1e-4,
                            lr_decay=0.1,
                            full=False,
                            fc_layers=[2_000])



va_transform = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
transform_map = dict(train=va_transform,
                    valid=va_transform,
                    test=va_transform)


data_cfg = ImageDatasetCfg(batch_size=512,
                            num_workers=4,
                            data_dir="/scratch/ssd004/scratch/jackk/birds_data",
                            device="cuda:0")


dsets = {split: ImageDataset(split, transform_map[split], data_cfg, ddp=False) for split in ["train", "valid", "test"]}

net = network.OurEfficientNet(efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1),
                        "/scratch/ssd004/scratch/jackk/birds_data/efficientnet_birds.dict", 
                        exp_cfg, dsets['train'].num_classes)
net.load_model_state_dict()
net.to("cuda:0")
net = hooks.GuidedBackprop(net)


default_scales = [3,5,7,9,13,15]
# pca_dirs = visualizations.find_pca_directions(dsets['train'], 8192, default_scales, 1)

# with open("big_sample_pca_dirs.pkl", "wb") as p:
#     pickle.dump(pca_dirs, p)

with open("big_sample_pca_dirs_reshaped.pkl", "rb") as p:
    pca_dirs = pickle.load(p)

for component in range(4):
    for sample in range(10):
        img,target_class,rand_idx= dsets['valid'].generate_one()
        print(sample, target_class, rand_idx)
        pca_saliency = visualizations.pca_direction_grids(net, dsets['valid'], target_class, img, default_scales, pca_dirs,
                                                          strides=2, component=component, batch_size=64)
        with open(f"./bird_pca_silu/guided_example_{component}_{sample}_{target_class}-{rand_idx}", "wb") as p:
            pickle.dump(pca_saliency, p)