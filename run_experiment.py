import numpy as np
import torch
from conv_color.training import run_experiments, run_experiment
from conv_color.color_regions import ColorDatasetGenerator, hard_color_classifier
from conv_color.config_objects import ColorDatasetConfig, ExperimentConfig
import os

torch.backends.cudnn.benchmark = True
device = "cuda:0" if torch.cuda.is_available() else "cpu"


train_indices = (0, 250_000) # size of training set
valid_indices = (1_250_000, 1_275_000)
test_indices = (3_260_000, 3_560_000)

critical_color_values = list(range(0,241,30))

dset_config = ColorDatasetConfig(task_difficulty="hard",
                                noise_size=(1,9),
                                num_classes=3,
                                num_objects=1,  # 0 => permuted, >0 => regular
                                radius=(1/8., 1/7.),
                                device=device,
                                batch_size=128)

# copies the config each time
train_set = ColorDatasetGenerator(train_indices, dset_config)
valid_set = ColorDatasetGenerator(valid_indices, dset_config)
test_set = ColorDatasetGenerator(test_indices, dset_config)
# train_set.cfg.infinite = True

# can specify a probability via the second value in the tuple for each entry  # initial hyperparam search
# hyperparameters=dict(learn_rate=[1e-4, 1e-3, 1e-2],
#                       weight_decay=10**np.linspace(-7, -1, 20),
#                       global_avg_pooling=[True, False],
#                       layer_sizes=dict(medium_size=[[16, 3, 1], [32, 3, 1]],
#                                        tiny_size=[[2, 3, 4], [6, 3, 4]],
#                                        large_size=[[16, 3, 1], [32, 3, 2], [32, 3, 2], [64, 3, 2]],
#                                        huge_size=[[16, 3, 1], [32, 3, 2], [32, 3, 2], [64, 3, 2], [64, 3, 1], [128, 3, 1]]),
#                     gain=[0, 0.05, 0.1, 0.2, 0.3],
#                     epochs=[25])
# prob_dists = dict(layer_sizes=[1.0, 0.0, 0.0, 0.0])
# run_experiments(train_set, valid_set, "small_circles_unpermuted", hyperparameters,
#                 search_type="random", prob_dists=prob_dists, num_rand=500)

# import pickle
# import dataclasses


# with open("./full_random_noisy/log.pkl", "rb") as p:
#     data = pickle.load(p)

# inv_layer_sizes = {str(v): k for k,v in hyperparameters["layer_sizes"].items()}

# best_run = {model_size:None for model_size in hyperparameters["layer_sizes"]}
# best_acc = {model_size:-float("inf") for model_size in hyperparameters["layer_sizes"]}
# for k,v in data["train_results"].items():
#     conf = dataclasses.asdict(k)
#     #conf["layer_sizes"] = inv_layer_sizes[str(conf["layer_sizes"])]
#     size_type = inv_layer_sizes[str(conf["layer_sizes"])]
#     if data["test_results"][k][1] > best_acc[size_type]:
#         best_acc[size_type] = data["test_results"][k][1]
#         best_run[size_type] = k
    
# model_name = "_".join(f"{{{k}}}" for k in hyperparameters) + ".dict"
# model_path = os.path.join("./full_random_noisy", model_name)
# for run_type, exp_config in best_run.items():
#     if run_type == "medium_size":
#         continue
#     dict_params = {}
#     for k,v in hyperparameters.items():
#         if isinstance(v, dict):
#             dict_params[k] = run_type 
#         else:
#             dict_params[k] = getattr(exp_config, k)


#     fetch_model_name = model_path.format(**dict_params)
#     exp_config.epochs = 10

#     print(run_experiment(train_set, valid_set, fetch_model_name, exp_config, extend=True))
extend_experiment = ExperimentConfig(learn_rate=1e-5,  # divide LR by 10x
                      weight_decay=8.858667904100833e-07,
                      global_avg_pooling=True,
                      layer_sizes=[[16, 3, 1], [32, 3, 1]],
                    epochs=15)
best_model = "small_circles_unpermuted/0.0001_8.858667904100833e-07_True_medium_size_0.1_25.dict"
print(run_experiment(train_set, valid_set, best_model, extend_experiment, extend=True))
