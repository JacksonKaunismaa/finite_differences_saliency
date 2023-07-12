import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os

from . import config_objects

class ConvBlock(nn.Module):
    def __init__(self, inpt_shape, layer_params, groups, act_func):
        super().__init__()
        lp = layer_params
        if lp[2] > 1: # stride
            pad_type = "valid"
            img_size = (inpt_shape[0]-lp[1])//lp[2] + 1 # https://arxiv.org/pdf/1603.07285.pdf
        else:
            img_size = inpt_shape[0]
            pad_type = "same"
        if isinstance(lp[0], float):
            lp[0] = int(lp[0])
            lp[0] -= lp[0] % groups # ensure divisble by groups
        self.is_resid = lp[2] == 1 and inpt_shape[-1] == lp[0]
        self.conv1 = nn.Conv2d(inpt_shape[2], lp[0], lp[1], stride=lp[2], padding=pad_type, groups=groups)
        self.batch_norm1 = nn.BatchNorm2d(lp[0]) # cant use track_running_stats=False since
        self.batch_norm2 = nn.BatchNorm2d(lp[0]) # it causes poor performance for inference with batch size=1 (or probably with the same image repeated a bunch of times)
        self.conv2 = nn.Conv2d(lp[0], lp[0], lp[1], stride=1, padding="same", groups=groups)
        self.act_func1 = getattr(torch.nn, act_func)()
        self.act_func2 = getattr(torch.nn, act_func)()

        self.output_shape = (img_size, img_size, lp[0])


    def forward(self, x):
        x_conv1 = self.act_func1(self.batch_norm1(self.conv1(x)))
        x_conv2 = self.act_func2(self.batch_norm2(self.conv2(x_conv1)))
        if self.is_resid:
            x = x + x_conv2  # residual block
        else:
            x = x_conv2  # dimension increasing block
        return x

class FullyConnectedBlock(nn.Module):
    def __init__(self, is_last, act_func, input_logits, outpt_logits):
        super().__init__()
        self.fully_connected = nn.Linear(input_logits, outpt_logits)
        # not sure why no batchnorms here?
        if is_last:
            self.act_func = nn.Identity()
        else:
            self.act_func = getattr(torch.nn, act_func)()

    def forward(self, x):
        return self.act_func(self.fully_connected(x))

class ResNet(nn.Module):
    def __init__(self, path, exp_config: config_objects.ExperimentConfig, dset_config: config_objects.ColorDatasetConfig):
        super().__init__()
        self.path = path
        self.cfg = exp_config
        self.dset_config = dset_config
        self.best_loss = float("inf")
        self.reload_cfg()

    def reload_cfg(self):  # build model layers from an update cfg attribute
        conv_blocks = []
        img_shape = self.dset_config.img_shape
        for ly in self.cfg.layer_sizes:  # (out_channels, kernel_size, stride) is each l
            conv_blocks.append(ConvBlock(img_shape, ly, self.cfg.groups, "ReLU"))
            img_shape = conv_blocks[-1].output_shape
        self.conv_blocks = nn.ModuleList(conv_blocks)

        self.final_num_logits = img_shape[0] * img_shape[1] * img_shape[2]
        self.final_img_shape = img_shape  # HWC

        fully_connected = []
        fc_layers = self.cfg.fc_layers[:]
        if self.cfg.global_avg_pooling:
            fc_layers.insert(0, img_shape[-1])
        else:
            fc_layers.insert(0, self.final_num_logits)
        fc_layers.append(self.dset_config.num_classes)

        for i, (fc_prev, fc_next) in enumerate(zip(fc_layers, fc_layers[1:])):
            is_last = i == len(fc_layers) - 2
            fully_connected.append(FullyConnectedBlock(is_last, "ReLU", fc_prev, fc_next))
        self.final_hidden = fc_layers[-1] if fc_layers else None
        self.fully_connected = nn.ModuleList(fully_connected)

    def forward(self, x, logits=False):
        for block in self.conv_blocks:
            x = block(x)

        if self.cfg.global_avg_pooling:
            x = torch.mean(torch.mean(x, -1), -1)
        else:
            x = torch.flatten(x, 1)
        for fc_layer in self.fully_connected:
            x = fc_layer(x)

        if self.dset_config.num_classes == 1 and not logits:  # always allow returning logits
            x = torch.sigmoid(x)
        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def try_convert_load(self, load_dict):
        new_state_dict = {}
        for k,v in load_dict["model"].items():
            if "conv_layers" in k:
                one_or_two, layer_num, param_type = k[11], k[13], k[15:]
                new_state_dict[f"conv_blocks.{layer_num}.conv{one_or_two}.{param_type}"] = v
            elif "batch_norms" in k:
                one_or_two, layer_num, param_type = k[11], k[13], k[15:]
                new_state_dict[f"conv_blocks.{layer_num}.batch_norm{one_or_two}.{param_type}"] = v
            elif "fully_connected" in k:
                layer_num, param_type = k[16], k[18:]
                new_state_dict[f"fully_connected.{layer_num}.fully_connected.{param_type}"] = v
        
        self.load_state_dict(new_state_dict)


    def maybe_save_model_state_dict(self, new_loss=None, new_acc=None, path=None, optim=None, sched=None):
        if new_loss is not None and new_loss >= self.best_loss:
            return
        
        if path is None:
            path = self.path

        #path = os.path.join("models", path)
        if new_loss is not None:
            self.best_loss = new_loss
        save_dict = {"model_sd": self.state_dict(),
                     "config": self.cfg,
                     #"dset_config": self.dset_config,
                     "best_loss": self.best_loss}    
        if optim is not None:
            save_dict["optim_sd"] = optim.state_dict()
        if sched is not None:
            save_dict["sched_sd"] = sched.state_dict()
        torch.save(save_dict, path)

    def load_model_state_dict(self, path=None, optim=None, sched=None):
        # convert old format
        if path is None:
            path = self.path
        if not os.path.exists(path):  # try checking models directory
            path = os.path.join("models", path)
            if not os.path.exists(path):
                print("No existing model found", path)
                return
        print("Found path of", path)
        load_dict = torch.load(path)

        if "config" in load_dict:  # else rely on client to supply correct config
            self.cfg = load_dict["config"]
            self.reload_cfg()

        if optim is not None:  # new_version: " optim_sd", old_version: "optim"
            opt_key = "optim_sd" if "optim_sd" in load_dict else "optim"
            if isinstance(load_dict[opt_key], tuple):  # fix for a typo we made earlier
                optim.load_state_dict(load_dict[opt_key][0])
            else:
                optim.load_state_dict(load_dict[opt_key])
        
        if sched is not None:
            sched.load_state_dict(load_dict["sched_sd"])

        try:
            if "model_sd" in load_dict: # new version
                self.load_state_dict(load_dict["model_sd"])
            elif "model" in load_dict:  # old version
                self.load_state_dict(load_dict["model"])

        except RuntimeError:  # need to convert old state_dict to new state_dict format
            # [print(k, v.shape, self.state_dict()[k].shape) for k,v in load_dict["model"].items()]
            # print(len(self.state_dict()))
            # print(len(load_dict["model"]))
            self.try_convert_load(load_dict)

        if "best_loss" in load_dict:  # some models were saved before this was added
            self.best_loss = load_dict["best_loss"]
            print("Model best_loss was", self.best_loss)
