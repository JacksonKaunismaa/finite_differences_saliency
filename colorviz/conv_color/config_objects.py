from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class ExperimentConfig:    
    # Architecture params
    layer_sizes: List = field(default_factory=list)
    fc_layers: List = field(default_factory=list)
    groups: int = 1
    global_avg_pooling: bool = False

    # Training params
    lr_decay: float = 0.1  # how much to decay after each milestone
    lr_max: float = 1e-3   # lr to start off at
    momentum: float = 0.9  # momentum
    step_size: int = 7   # how often to drop lr
    weight_decay: float = 1e-4
    epochs: int = 250
    full: bool = False  # whether or not to finetune the full model

    def __hash__(self):
        return hash(str(self))


@dataclass
class ColorDatasetConfig:
    # noise parameters
    num_noise: Tuple = (50, 70)  # number of locations to generate noise at
    noise_size: Tuple = (1, 15)  # size that each noise instance can be

    # Image parameters
    size: int = 128 # HW dimension of images
    channels: int = 1  # default is greyscale
    img_shape = (size, size, channels)  # HWC shape of input images
    bg_color: int = 0  # must have same number of channels

    # Target parameters
    task_difficulty: str = None  # function that maps colors to classes (supports iterables)
    num_classes: int = 2  # how many possible classes there are
    radius: Tuple[float, float] = (1/6., 1/3.)  # range of possible radii for circles as a fraction of size
    num_objects: int = 1 # supports ranges, if want multiclass
    overlap_attempts: int = 0  # if >0 will attempt to generate circles overlap_attempts times before giving up. if ==0 then overlaps allowed
    # Greyscale
    color_range: Tuple = (5, 255)  # range of values that the greyscale color-to-be-classified can be
    # RGB (which we generate as HSV for simplicity)
    value_range: Tuple  = (20, 100) # range for value in HSV (subset of (0, 100))
    saturation_range: Tuple = (20, 100) # range for saturation in HSV (subset of (0, 100))
    hue_range: Tuple = (0, 360)  # range for hue in HSV (subset of (0, 360))

    # Dataset Parameters
    # controlled directly by the Dataset now
    infinite = False  # if true, will not restrict itself to indices in image_indices (but will be limited lengthwise to that).
    device: str = "cpu"  # where dataloaders should get sent to
    permute_seed: int = 0  # for permuted pixels, seeds used to create the fixed permutations
    batch_size: int = 32
    num_workers: int = 6


@dataclass
class ImageDatasetCfg:
    batch_size: int = 32
    num_workers: int = 4
    data_dir: str = ""
    device: str = ""
