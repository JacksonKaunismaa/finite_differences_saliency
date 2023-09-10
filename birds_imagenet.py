from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, EfficientNet
import torch.nn as nn
import torch
import wandb
import dataclasses
from torchvision.transforms import transforms
import types
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import os

from colorviz.birds_dataset.data import ImageDataset
from colorviz.conv_color.config_objects import ImageDatasetCfg, ExperimentConfig
from colorviz.conv_color import training
from colorviz.conv_color import utils
from colorviz.birds_dataset import network

torch.backends.cudnn.benchmark = True

def main(rank, args):
    is_ddp = rank is not None
    if rank is None:
        rank = 0
    if is_ddp:
        dist.init_process_group(backend="nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)  # so that nccl knows we are only using that specific device
        print("hi from", rank)
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    # lr schedule
    # simplify the task
    # lower the learning rate
    tr_transform = transforms.Compose([transforms.Resize(256, antialias=True),
                                    transforms.RandomCrop(224),
                                    #transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ???
                                                         std=[0.229, 0.224, 0.225]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(8.),
                                    transforms.ColorJitter(contrast=0.1)
                                                         ])


    # load_model_state_dict(net, optim=opt, sched=combined_sched)


    va_transform = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
    transform_map = dict(train=tr_transform,
                         valid=va_transform,
                         test=va_transform)


    data_cfg = ImageDatasetCfg(batch_size=1792//torch.cuda.device_count(),
                                num_workers=4,
                                data_dir="/scratch/ssd004/scratch/jackk/birds_data",
                                device=device)
    

    dsets = {split: ImageDataset(split, transform_map[split], data_cfg, is_ddp) for split in ["train", "valid", "test"]}

    exp_cfg = ExperimentConfig(epochs=30,
                                lr_max=4e-3,
                                step_size=7,
                                weight_decay=1e-4,
                                lr_decay=0.1,
                                full=False,
                                fc_layers=[2_000])

    net = network.OurEfficientNet(efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1),
                        "./checkpoint/efficientnet_birds.dict", exp_cfg, dsets['train'].num_classes)

    # net.eval() # set batchnorms to inference mode (this would just get erased during evaluation anyway)
    if isinstance(net.model.classifier, nn.Sequential):
        print("starting classifier mean norm per logit:", net.model.classifier[1][-1].weight.norm(dim=1).mean())
    else:
        print("starting classifier mean norm per logit:", net.model.classifier[1].weight.norm(dim=1).mean())


    print("Trainable params:", [name for name,param in net.named_parameters() if param.requires_grad])
    print(net.model.features[7][0].block[0][1], 
          net.model.features[7][0].block[0][1].training, 
          net.model.features[7][0].block[0][1].track_running_stats)

    if utils.get_rank() == 0:
        print(net)

    net = net.to(device)
    if is_ddp:
        net = DDP(net, device_ids=[rank], output_device=rank)    
    net = torch.compile(net)
    raw_net = utils.get_raw(net)


    optimizer = torch.optim.Adam([param for param in net.parameters() if param.requires_grad],
                                lr=exp_cfg.lr_max,
                                weight_decay=exp_cfg.weight_decay)
                                #momentum=raw_net.cfg.momentum)

    scheduler = StepLR(optimizer, step_size=exp_cfg.step_size, gamma=exp_cfg.lr_decay)


    if utils.get_rank() == 0:
        wandb.init(
            project="birds_dataset",
            config={"name": raw_net.path,
                    "job_id": os.environ.get("SLURM_JOB_ID", -1),
                    **dataclasses.asdict(data_cfg),
                    **dataclasses.asdict(exp_cfg)},
        )
    training.train(net, optimizer, nn.CrossEntropyLoss(), exp_cfg.epochs, dsets, scheduler=scheduler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world-size', default=1, type=int, help="num gpu processes (limited to 1 node)")
    args = parser.parse_args()
    if args.world_size > 1:
        mp.spawn(main, args=(args,), nprocs=args.world_size) # type: ignore
    else:
        main(None, args)
















# perturb test images  (of course it didnt work)
# test nanning, see if problem has grown (probably has) -> ask question about relative position encoding
