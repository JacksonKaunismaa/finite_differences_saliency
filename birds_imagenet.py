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

    class OurEfficientNet(nn.Module):
        def __init__(self, net: EfficientNet, path: str, exp_cfg: ExperimentConfig, n_classes: int):
            super().__init__()
            self.model = net

            if not exp_cfg.full:
                for param in self.model.parameters():
                    param.requires_grad = False
            
            self.path = path
            self.cfg = exp_cfg
            self.best_acc = float("-inf")

            if exp_cfg.fc_layers:
                net.classifier[1] = nn.Sequential(nn.Linear(net.classifier[1].in_features, exp_cfg.fc_layers[0]),
                                                nn.ReLU(),
                                                nn.Linear(exp_cfg.fc_layers[0], n_classes)
                                                )
            else:
                net.classifier[1] = nn.Linear(net.classifier[1].in_features, n_classes)

        def forward(self, x):
            return self.model(x)

        def maybe_save_model_state_dict(self, new_loss=None, new_acc=None, path=None, optim=None, sched=None):
            if new_acc is not None and new_acc <= self.best_acc:
                return

            if path is None:
                path = self.path

            #path = os.path.join("models", path)
            if new_acc is not None:
                self.best_acc = new_acc
            save_dict = {"model_sd": self.state_dict(),
                            "config": self.cfg,
                            #"dset_config": self.dset_config,
                            "best_acc": self.best_acc}
            if optim is not None:
                save_dict["optim_sd"] = optim.state_dict()
            if sched is not None:
                save_dict["sched_sd"] = sched.state_dict()
            torch.save(save_dict, path)

        def load_model_state_dict(net, optim=None, sched=None):
                print("Found path of", net.path)
                load_dict = torch.load(net.path)

                if optim is not None:
                    optim.load_state_dict(load_dict["optim_sd"])
                if sched is not None:
                    sched.load_state_dict(load_dict["sched_sd"])

                net.load_state_dict(load_dict["model_sd"])

        def train(self, mode: bool=True):
            print(".train() called")
            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    print("set only this guy to", mode)
                    module.train(mode)
                else:  # ie. force all other layers to be in evaluation mode except for the classifier
                    module.train(False)

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

    net = OurEfficientNet(efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1),
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
