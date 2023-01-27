import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os

class ConvBlock(nn.Module):
    def __init__(self, inpt_shape, layer_params, groups, act_func):
        super().__init__()
        l = layer_params
        if l[2] > 1: # stride
            pad_type = "valid"
            img_size = (inpt_shape[0]-l[1])//l[2] + 1 # https://arxiv.org/pdf/1603.07285.pdf
        else:
            img_size = inpt_shape[0]
            pad_type = "same"
        if isinstance(l[0], float):
            l[0] = int(l[0])
            l[0] -= l[0] % groups # ensure divisble by groups
        self.is_resid = l[2] == 1 and inpt_shape[-1] == l[0]
        self.conv1 = nn.Conv2d(inpt_shape[2], l[0], l[1], stride=l[2], padding=pad_type, groups=groups)
        self.batch_norm1 = nn.BatchNorm2d(l[0]) # cant use track_running_stats=False since
        self.batch_norm2 = nn.BatchNorm2d(l[0]) # it causes poor performance for inference with batch size=1 (or probably with the same image repeated a bunch of times)
        self.conv2 = nn.Conv2d(l[0], l[0], l[1], stride=1, padding="same", groups=groups)
        self.act_func1 = getattr(torch.nn, act_func)()
        self.act_func2 = getattr(torch.nn, act_func)()
        
        self.output_shape = (img_size, img_size, l[0])
    
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
    def __init__(self, conv_layers, num_classes, img_shape, path, fc_layers=[1000], groups=1):
        super().__init__()
        conv_blocks = []
        self.path = path
        self.num_classes = num_classes
        for l in conv_layers:  # (out_channels, kernel_size, stride) is each l
            conv_blocks.append(ConvBlock(img_shape, l, groups, "ReLU"))
            img_shape = conv_blocks[-1].output_shape
        self.conv_blocks = nn.ModuleList(conv_blocks)

        self.final_num_logits = img_shape[0] * img_shape[1] * img_shape[2]
        fully_connected = []
        fc_layers.insert(0, self.final_num_logits)
        fc_layers.append(num_classes)
        for i, (fc_prev, fc_next) in enumerate(zip(fc_layers, fc_layers[1:])):
            is_last = i == len(fc_layers) - 2
            fully_connected.append(FullyConnectedBlock(is_last, "ReLU", fc_prev, fc_next))
        self.fully_connected = nn.ModuleList(fully_connected)

    def forward(self, x, logits=False):
        for block in self.conv_blocks:
            x = block(x)
        x = torch.flatten(x, 1)
        for fc_layer in self.fully_connected:
            x = fc_layer(x)
            
        if self.num_classes == 1 and not logits:  # always allow returning logits
            x = torch.sigmoid(x)
        return x    

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_model_state_dict(self, path=None, optim=None):
        if path is None:
            path = self.path
        if optim is not None:
            save_dict = {}
            save_dict["model"] = self.state_dict()
            save_dict["optim"] = optim.state_dict()
        else:
            save_dict = self.state_dict()
        torch.save(save_dict, path)
    
    def load_model_state_dict(self, path=None, optim=None):
        # convert old format
        if path is None:
            path = self.path
        if not os.path.exists(path):
            return
        load_dict = torch.load(path)
        
        if "model" in load_dict:
            if optim is not None:
                optim.load_state_dict(load_dict["optim"])
            try:
                self.load_state_dict(load_dict["model"])
            except RuntimeError:  # need to convert old state_dict to new state_dict format
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
        else:
            self.load_state_dict(load_dict)

def correct(pred_logits, labels):
    if labels.shape[1] != 1:
        pred_probabilities = F.softmax(pred_logits, dim=1)
        classifications = torch.argmax(pred_probabilities, dim=1)
        labels_argmax = torch.argmax(labels, dim=1)
    else:
        classifications = pred_logits.int()
        labels_argmax = labels
    correct = (labels_argmax == classifications)
    return correct

def train(net, optimizer, loss, epochs, device=None):
    va_losses = []
    tr_losses = []
    va_accuracies = []
    for epoch in range(epochs):
        epoch_tr_loss = 0.0
        net.train()
        for i, sample in tqdm(enumerate(train_loader)):
            imgs = sample["image"].to(device, non_blocking=False).float()
            labels = sample["label"].to(device).float()
            optimizer.zero_grad()
            outputs = net(imgs)
            batch_loss = loss(outputs, labels)
            epoch_tr_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
        epoch_va_loss = 0.0
        epoch_va_correct = 0
        net.eval()
        with torch.no_grad():
            for i, sample in enumerate(valid_loader):
                imgs = sample["image"].to(device).float()
                labels = sample["label"].to(device).float()
                outputs = net(imgs)
                epoch_va_loss += loss(outputs, labels).item()
                epoch_va_correct += correct(outputs, labels).sum().item()
        epoch_va_accuracy = epoch_va_correct/(valid_indices[1] - valid_indices[0])
        print(f'Epoch {epoch + 1}: va_loss: {epoch_va_loss}, va_accuracy: {epoch_va_accuracy}, tr_loss: {epoch_tr_loss}')
        if not va_losses or epoch_va_loss < min(va_losses):
            net.save_model_state_dict(optim=optimizer)
        va_losses.append(epoch_va_loss)
        tr_losses.append(epoch_tr_loss)
        va_accuracies.append(epoch_va_accuracy)
    return va_losses, va_accuracies, tr_losses
