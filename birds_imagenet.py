from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
import torch
import wandb
import dataclasses
from torchvision.transforms import transforms
import types
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, MultiplicativeLR, SequentialLR


from colorviz.birds_dataset.data import ImageDataset
from colorviz.conv_color import config_objects
from colorviz.conv_color import training

device = "cuda:0" if torch.cuda.is_available() else "cpu"

net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
print(net)

data_cfg = config_objects.ImageDatasetCfg(batch_size=32,
                               num_workers= 4,
                               data_dir="./bird_data",
                               device=device)

# transform = transforms.Compose([transforms.Resize(256, antialias=True), 
#                                 transforms.CenterCrop(224),
#                                 #transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ???
#                                                      std=[0.229, 0.224, 0.225])]
# )
transform = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
dsets = {split: ImageDataset(split, transform, data_cfg) for split in ["train", "valid", "test"]}
net.cfg = config_objects.ExperimentConfig(epochs=300, 
                                          lr_min=1e-1,
                                          lr_max=1.,
                                          t_warmup=40,
                                          fc_layers=[2_000])

net.classifier[1] = nn.Sequential(nn.Linear(net.classifier[1].in_features, net.cfg.fc_layers[0], device=device),
                                  nn.ReLU(),
                                  nn.Linear(net.cfg.fc_layers[0], dsets["train"].num_classes)
                                  )
net = net.to(device)
net.best_acc = float("-inf")

for name,param in net.named_parameters():
    if "classifier" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

print("Trainable params:", [name for name,param in net.named_parameters() if param.requires_grad])
opt = torch.optim.Adam([param for param in net.parameters() if param.requires_grad], lr=net.cfg.lr_max)
warmup_sched = LinearLR(opt, start_factor=net.cfg.lr_min/net.cfg.lr_max,
                        end_factor=1., total_iters=net.cfg.t_warmup)
# t_decay - t_warmup since we include the first t_warmup iters in the "decay" period
const_sched = MultiplicativeLR(opt, lr_lambda=lambda step: 1)

combined_sched = SequentialLR(opt, [warmup_sched, const_sched],
                                      milestones=[net.cfg.t_warmup])


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


net.path = "models/high_lr_multi_fc_efficientnet_birds_warmup.dict"
# load_model_state_dict(net, optim=opt, sched=combined_sched)
net.maybe_save_model_state_dict = types.MethodType(maybe_save_model_state_dict, net)

wandb.init(
    project="birds_dataset",
    config={"name": net.path, **dataclasses.asdict(data_cfg)},
    # id="gknaxb9f",
    # resume="must"
)
training.train(net, opt, nn.CrossEntropyLoss(), net.cfg.epochs, dsets, scheduler=combined_sched)
















# perturb test images  (of course it didnt work)
# test nanning, see if problem has grown (probably has) -> ask question about relative position encoding
