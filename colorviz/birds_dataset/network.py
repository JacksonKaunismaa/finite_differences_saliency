import torch.nn as nn
from torchvision.models import EfficientNet
import torch

from ..conv_color.config_objects import ExperimentConfig


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
        # print(".train() called")
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                # print("set only this guy to", mode)
                module.train(mode)
            else:  # ie. force all other layers to be in evaluation mode except for the classifier
                module.train(False)