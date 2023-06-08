import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import itertools
import os
import config_objects
import network
import hooks
import numpy as np
import wandb
import dataclasses

def run_experiment(train_dset, valid_dset, name, exp_config: config_objects.ExperimentConfig, extend=False):
    wandb.init(
        project=os.path.dirname(name),
        config=dataclasses.asdict(exp_config),
    )
    net = network.ResNet(name, exp_config, train_dset.cfg).to(train_dset.cfg.device)
    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=exp_config.learn_rate, weight_decay=exp_config.weight_decay)
    if extend:
        net.load_model_state_dict(optim=optim)
        net.to(train_dset.cfg.device)
    if exp_config.gain and not extend:
        hooks.set_initializers(net, exp_config.gain)
    train_result = train(net, optim, loss_func, exp_config.epochs, train_dset, valid_dset)
    net.load_model_state_dict()  # load the best model
    net = net.to(train_dset.cfg.device)
    test_result = evaluate(net, loss_func, valid_dset)
    wandb.finish()

    return train_result, test_result


def run_experiments(train_dset, valid_dset, experiment_path, hyperparams, prob_dists=None, search_type="grid", override=False, num_rand=20):
    assert search_type in ["grid", "random"]
    assert all(k in config_objects.ExperimentConfig().__dataclass_fields__ for k in hyperparams)
    if prob_dists is None:
        prob_dists = {}

    os.makedirs(experiment_path, exist_ok=True)

    log_path = os.path.join(experiment_path, "log.pkl")
    if os.path.exists(log_path):
        with open(log_path, "rb") as p:
            saved = pickle.load(p)
        if saved["dset_cfg"] != train_dset.cfg: #or saved["hyperparams"] != hyperparams:
            print("Found existing log at path with different config than specified")
            if not override:  # override: set this to ignore the dset_config equality check
                return
        train_results, test_results = saved["train_results"], saved["test_results"]
    else:
        train_results, test_results = {}, {}

    model_name = "_".join(f"{{{k}}}" for k in hyperparams) + ".dict"
    model_path = os.path.join(experiment_path, model_name)

    log_dict = {"hyperparams": hyperparams,
                "dset_cfg": train_dset.cfg,
                "train_results": train_results,
                "test_results": test_results}

    hyp_keys, hyp_choices = list(hyperparams.keys()), list(hyperparams.values())
    experiment_iter = itertools.product(*hyp_choices) if search_type == "grid" else range(num_rand)
    for i, item in enumerate(experiment_iter):
        if search_type == "grid":  # here, "item" is the actual selections for the hyperparameters
            hyp_dict = dict(zip(hyp_keys, item))
        elif search_type == "random":  # here "item" is just an integer
            hyp_dict = {}
            for k,choices in hyperparams.items():
                prob_dist = prob_dists.get(k)
                if isinstance(choices, dict):
                    choices = list(choices.keys())                
                hyp_dict[k] = np.random.choice(choices, p=prob_dist)
        
        # use the named choices version of hyp_dict
        name = model_path.format(**hyp_dict)  # guarantees that the format specified in name matches the actual hyperparams

        # fetch the values associated with the named choices
        for k,choice in hyp_dict.items():
            if isinstance(choice, str):  # assume that if a hyperparameter takes a string value, it's a named choice
                hyp_dict[k] = hyperparams[k][choice]

        # use the value-only version of hyp_dict
        exp_config = config_objects.ExperimentConfig(**hyp_dict)
        if exp_config in train_results:
            print("Already completed experiment for", name)
            continue
        print("Running experiment for", name, "experiment", i+1)

        train_result, test_result = run_experiment(train_dset, valid_dset, name, exp_config)
        
        train_results[exp_config] = train_result
        test_results[exp_config] = test_result

        with open(log_path, "wb") as p:
            pickle.dump(log_dict, p)
            

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


@torch.no_grad()
def evaluate(net, loss, dataset):
    epoch_va_loss = 0.0
    epoch_va_correct = 0
    net.eval()
    total_valid = 0
    for i, sample in enumerate(dataset.dataloader()):
        imgs = sample["image"].to(dataset.cfg.device).float()
        labels = sample["label"].to(dataset.cfg.device).float()
        outputs = net(imgs)
        epoch_va_loss += loss(outputs, labels).item()
        epoch_va_correct += correct(outputs, labels).sum().item()
        total_valid += labels.shape[0]
    epoch_va_accuracy = epoch_va_correct/total_valid
    return epoch_va_loss, epoch_va_accuracy


def train(net, optimizer, loss, epochs, train_set, valid_set, log_file=None,
          track_stat=None, summarize=None, test_set=None, verbose=True):
    va_losses = []
    tr_losses = []
    extra_stats = []
    summary_stats = []
    va_accuracies = []

    for epoch in range(epochs):
        epoch_tr_loss = 0.0
        net.train()
        #input("Entering train")
        for i, sample in tqdm(enumerate(train_set.dataloader())):
            imgs = sample["image"].to(train_set.cfg.device, non_blocking=False).float()
            labels = sample["label"].to(train_set.cfg.device).float()
            outputs = net(imgs)
            batch_loss = loss(outputs, labels)
            epoch_tr_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        epoch_va_loss, epoch_va_accuracy = evaluate(net, loss, valid_set)
        epoch_summary = f'Epoch {epoch + 1}: va_loss: {epoch_va_loss}, va_accuracy: {epoch_va_accuracy}, tr_loss: {epoch_tr_loss}'
        #input("Entering extra")
        if track_stat is not None:
            track_dataset = test_set if test_set is not None else valid_set
            curr_stat = track_stat(net, loss, optimizer, track_dataset)
            if summarize is not None:
                summary_stat = summarize(curr_stat)
            else:
                summary_stat = curr_stat
            extra_stats.append(curr_stat)
            summary_stats.append(summary_stat)
            epoch_summary += f", curr_stat: {summary_stat}"
        if verbose:
            print(epoch_summary)
        if epoch_va_loss < net.best_loss:
            net.save_model_state_dict(epoch_va_loss, optim=optimizer)
        wandb.log({"va_loss": epoch_va_loss, "va_acc": epoch_va_accuracy, "tr_loss": epoch_tr_loss})
        va_losses.append(epoch_va_loss)
        tr_losses.append(epoch_tr_loss)
        va_accuracies.append(epoch_va_accuracy)
        if log_file is not None:
            with open(log_file, "wb") as p:
                pickle.dump((va_losses, va_accuracies, tr_losses, extra_stats, summary_stats), p)
    return va_losses, va_accuracies, tr_losses, extra_stats, summary_stats
