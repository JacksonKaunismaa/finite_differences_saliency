import torch
from torch.utils.data import Dataset
import numpy as np
import colorsys
# import imageio
import scipy.ndimage
import os
from tqdm import tqdm
import copy
from torchvision import transforms

from . import config_objects

def hard_color_classifier(color):  
    if color <= 30:  # 3 classes
        return 0
    if 30 < color <= 60:  # => 90/255 is 0, 90/255 is 1, 75/255 is 2
        return 1
    if 60 < color <= 90:
        return 2
    if 90 < color <= 120:
        return 1
    if 120 < color <= 150:
        return 0
    if 150 < color <= 180:
        return 1
    if 180 < color <= 210:
        return 2
    if 210 < color <= 240:
        return 0
    if 240 < color:
        return 2
    
def medium_color_classifier(color):
    if color <= 100:  
        return 0
    if 100 < color <= 150:
        return 1
    if 150 < color <= 200: 
        return 2
    if 200 < color:
        return 1
    
class ColorDatasetGenerator(Dataset):
    def __init__(self, image_indices, config: config_objects.ColorDatasetConfig):
        self.cfg = copy.copy(config)
        self.image_indices = image_indices
        self.transform = transforms.Compose([transforms.ToTensor()])  # default transform, cant put it in config 

        # design pattern: any variables that can be derived from other values in config should not be added to config, and
        #               don't put any Callables in them either
        color_tasks = {"hard": hard_color_classifier, "medium": medium_color_classifier}
        self.color_classifier = color_tasks[config.task_difficulty]
    
        if self.cfg.num_classes == 2:
            self.class_multiplier = lambda target_class: 1 if target_class == 1 else -1
        
        self.actual_radii = (int(self.cfg.radius[0]*self.cfg.size), int(self.cfg.radius[1]*self.cfg.size))
        self.rng = np.random.RandomState(self.cfg.permute_seed)  # use for setting seed so that we don't affect global seed

        # ability to choose between several pre-set permutations
        self.random_permutes = [np.mgrid[:self.cfg.size, :self.cfg.size].transpose(1,2,0).reshape(-1,2) for _ in range(30)]
        for perm in self.random_permutes:
            self.rng.shuffle(perm) 
        self.random_permutes = [x.reshape(self.cfg.size, self.cfg.size, 2) for x in self.random_permutes]  

    def iterative_color_cvt(self, conversion, iterable):  # covert between hsv and csv
        # iterates over first dimension
        convert_func = getattr(colorsys, conversion)
        return np.array(list(map(lambda x: convert_func(*x), iterable)))

    def generate_colors(self, amt):
        if amt == 0:
            amt = 1
        if self.cfg.channels == 1: # greyscale
            color = self.rng.randint(*self.cfg.color_range, (amt))
        else:  # rgb
            hue = self.rng.randint(*self.cfg.hue_range, (amt))/360
            saturation = self.rng.randint(*self.cfg.saturation_range, (amt))/100
            value = self.rng.randint(*self.cfg.value_range, (amt))/100
            color = (self.iterative_color_cvt("hsv_to_rgb", zip(hue, saturation, value))*255.).round() #(np.array(list(map(lambda x: colorsys.hsv_to_rgb(*x), zip(hue, saturation, value))))*255).round()
        return color

    def get_radii_locations(self, num_objects):
        if self.cfg.overlap_attempts > 0 and num_objects > 1:
            radii = [self.rng.randint(*self.actual_radii)]
            locations = [self.rng.randint(self.actual_radii[1], self.cfg.size-self.actual_radii[1], (1, 2))]
            for _ in range(num_objects-1):
                for i in range(self.cfg.overlap_attempts):  # overlap_attempts attempts to make them non-overlapping
                    next_radius = self.rng.randint(*self.actual_radii)
                    next_loc = self.rng.randint(self.actual_radii[1], self.cfg.size-self.actual_radii[1], (1, 2))
                    distances = np.linalg.norm(next_loc-locations) - (radii + next_radius)  # then they are too close, so try again
                    if np.any(distances < 0): # then they are too close
                        continue
                    break  # ie. we found one sufficiently far from everything
                else:  # if we reach the end, then at least for all 20 attempts, one was too close, so just skip
                    continue
                radii.append(next_radius)
                locations.append(next_radius)
        else:
            radii = self.rng.randint(*self.actual_radii, (num_objects))
            locations = self.rng.randint(self.actual_radii[1], self.cfg.size-self.actual_radii[1], (num_objects, 2))

        return radii, locations

    def add_target(self, arr, set_color):
        num_objects = self.cfg.num_objects # self.rng.randint(*self.num_objects)
        if num_objects == 0:
            num_objects = 1

        if set_color is not None:
            color = np.array([set_color]*num_objects)
        else:
            color = self.generate_colors(num_objects)

        label = np.zeros((self.cfg.num_classes))
        label[self.color_classifier(color)] = 1  # multi-hot encoded
        if self.cfg.num_classes == 2:
            label = np.expand_dims(label[0], 0)

        radii,locations = self.get_radii_locations(num_objects)

        for radius, location in zip(radii, locations):
            x_coords = np.arange(radius)
            for x in x_coords:
                height = 2*int(np.sqrt(radius**2 - x**2))
                y_coords = np.arange(height) - height//2 + location[1]
                arr[location[0]+x, y_coords] = color
                arr[location[0]-x, y_coords] = color

        return label, color, radii, locations  # doesnt really work if we are doing multiclass

    def add_noise(self, arr):
        num_noise = self.rng.randint(*self.cfg.num_noise)
        sizes = self.rng.randint(*self.cfg.noise_size, num_noise)
        colors = self.generate_colors(num_noise)
        locations = self.rng.randint(self.cfg.noise_size[1], self.cfg.size-self.cfg.noise_size[1], (num_noise, 2))
        for size, color, location in zip(sizes, colors, locations): # location is upper left corner
            arr[location[0]:location[0]+size,location[1]:location[1]+size] = color
        return sizes, colors, locations

    def generate_one(self, set_color=None):  # generating on the fly is faster than loading from disk
        img = np.ones((self.cfg.size, self.cfg.size, self.cfg.channels)) * self.cfg.bg_color
        label, color, size, pos = self.add_target(img, set_color)
        noise = self.add_noise(img)
        if self.cfg.num_objects == 0:
            permute_choice = self.rng.choice(len(self.random_permutes))
            img_perm = img[self.random_permutes[permute_choice][...,0], self.random_permutes[permute_choice][...,1]]
            return img_perm, label, color, size, pos, noise, img
        return img, label, color, size, pos, noise

    def __len__(self):
        return self.image_indices[1] - self.image_indices[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx += self.image_indices[0] # shift into dataset range

        if self.cfg.infinite: 
            self.rng = np.random  # for when results shouldn't be repeatable
        else:
            self.rng = np.random.RandomState(idx)  # to make results repeatable

        image, label, color, *_ = self.generate_one()
        if hasattr(self, "transform"):
            image = self.transform(image)
        sample = {'image': image, 'label': label, 'color': color, "seeds": idx}
        return sample
    
    def dataloader(self, shuffle=True):
        return torch.utils.data.DataLoader(self, batch_size=self.cfg.batch_size, 
                                          shuffle=shuffle, num_workers=self.cfg.num_workers, pin_memory=True)

    def implicit_normalization(self, inpt):
        # since ColorDatasetGenerator.generate_one returns np.float32, transforms.ToTensor() will implicitly assume we are already
        #in [0,1] (even though we aren't), and won't do a divide by 255. Since we work in [0, 255] space for basically everything, and
        # the ColorDatasetGenerator models take input in [0, 255] (due to ToTensor having assumed we were in [0,1]), we don't have to
        # do anything here, and only exists because TextureDatasetGenerator actually requires us to do something here
        return inpt



# class TextureDatasetGenerator(Dataset):
#     def __init__(self, dtd_loc, **kwargs):
#         super().__init__()
#         # define default values for parameters, can override any using kwargs
#         # noise parameters
#         self.num_noise = (50, 70)  # number of locations to generate noise at
#         self.noise_size = (5, 10)  # size that each noise instance can be
# #         self.value_range = (20, 100) # range for value in HSV (subset of (0, 100))
# #         self.saturation_range = (20, 100) # range for saturation in HSV (subset of (0, 100))
# #         self.hue_range = (0, 360)  # range for hue in HSV (subset of (0, 360))

#         # image parameters
#         self.size = 128 # shape of image
#         self.channels = 3  # default is greyscale
#         self.bg_color = 0  # must have same number of channels

#         # target parameters
#         self.num_classes = 2  # how many possible classes there are
#         self.textures = []  # list of texture images (order matters)
#         self.texture_labels = []  # list of class idxes that each texture is in
#         self.texture_file_names = []  # list of filename associated to each texture
#         self.texname_to_idx = {}  # maps texture types to their class indices
#         self.idx_to_texname = {}  # maps class indices to texture types

#         self.radius_frac = (1./6, 1./3)  # range of possible radii (fraction of self.size)
#         self.num_objects = 1 # supports ranges, if want multiclass

#         # actual dataset
#         #self.labels = {}  # {filename: label} mapping
#         self.image_indices = (0, 1_000_00)   # range of self.rng.seeds for the given dataset
#         self.options = []  # list of dataset options that need to be saved
#         # self.save_dir = ""   # location for all images to be stored
#         for k,v in kwargs.items():
#             setattr(self, k, v)
#             self.options.append(k)

#         if isinstance(dtd_loc, str):
#             self.load_dtd_textures(os.path.join(dtd_loc, "images"),
#                                 os.path.join(dtd_loc, "labels", "labels_joint_anno.txt"))
#         elif isinstance(dtd_loc, TextureDatasetGenerator):
#             for attrib in ["textures", "texture_file_names", "texture_labels", "num_classes", "idx_to_texname"]:
#                 setattr(self, attrib, getattr(dtd_loc, attrib))

#     def load_dtd_textures(self, images_path, labels_file):
#         with open(labels_file, "r") as f:
#             labels = f.readlines()
#         for label in tqdm(labels):
#             name, *categ = label.split()
#             if len(categ) > 1: # some textures are multi-class, ignore these for now
#                 continue
#             if categ[0] not in self.texname_to_idx:
#                 self.texname_to_idx[categ[0]] = len(self.texname_to_idx)
#             imread = imageio.v2.imread(os.path.join(images_path, name))

#             downsampled = scipy.ndimage.zoom(imread,
#                                             [self.size/imread.shape[0], self.size/imread.shape[1], 1.],
#                                             order=1)
#             self.textures.append(downsampled)
#             self.texture_file_names.append(name)
#             self.texture_labels.append(self.texname_to_idx[categ[0]])

#         self.idx_to_texname = {y:x for x,y in self.texname_to_idx.items()}
#         # self.textures = np.asarray(self.textures) # images have different shapes
#         self.texture_labels = np.asarray(self.texture_labels)
#         self.num_classes = len(self.texname_to_idx)

#     @property
#     def radius(self):
#         return int(self.radius_frac[0]*self.size), int(self.radius_frac[1]*self.size)

#     def add_target(self, arr, num_objects, textures):

#         # pick a random texture image
#         # pick a random location to sample that image at (since our image size is smaller than the texture image size)
#         #sample_locs_x = self.rng.randint(self.textures[textures[0]].shape[0]-self.size, size=num_objects)
#         #sample_locs_y = self.rng.randint(self.textures[textures[0]].shape[1]-self.size, size=num_objects)
#         label = np.zeros((self.num_classes))
#         label[self.texture_labels[textures[0]]] = 1  # multi-hot encoded
#         if self.num_classes == 2:
#             label = np.expand_dims(label[0], 0)

#         # probably should make sure they dont overlap too much, but num_objects=1 for now
#         radii = self.rng.randint(*self.radius, (num_objects))
#         locations = self.rng.randint(self.radius[1], self.size-self.radius[1], (num_objects, 2))
#         target_zip = zip(radii, locations, textures)#, sample_locs_x, sample_locs_y)
#         for radius, location, texture in target_zip:#, sample_loc_x, sample_loc_y in target_zip:
#             x_coords = np.arange(radius)
#             # subsampling of image guaranteed to be at least size x size
#             tex_image = self.textures[texture]#[sample_loc_x:, sample_loc_y:, :]
#             for x in x_coords:
#                 height = 2*int(np.sqrt(radius**2 - x**2))
#                 y_coords = np.arange(height) - height//2 + location[1]
#                 arr[location[0]+x, y_coords] = tex_image[location[0]+x, y_coords]
#                 arr[location[0]-x, y_coords] = tex_image[location[0]-x, y_coords]
#         return label, radii, locations  # doesnt really work if we are doing multiclass

#     def generate_colors(self, amt):
#         return self.rng.randint(0,255, size=(amt,3))

#     def add_noise(self, arr):
#         num_noise = self.rng.randint(*self.num_noise)
#         sizes = self.rng.randint(*self.noise_size, num_noise)
#         colors = self.generate_colors(num_noise)
#         locations = self.rng.randint(self.noise_size[1], self.size-self.noise_size[1], (num_noise, 2))
#         for size, color, location in zip(sizes, colors, locations):
#             arr[location[0]:location[0]+size,location[1]:location[1]+size] = color

#     def generate_one(self):
#         num_objects = self.num_objects # self.rng.randint(*self.num_objects)
#         tex_indices = self.rng.randint(len(self.textures), size=num_objects)
#         bg_image = self.rng.randint(len(self.textures))
#         while self.texture_labels[tex_indices[0]] == self.texture_labels[bg_image]:
#             bg_image = self.rng.randint(len(self.textures))
#         img = self.textures[bg_image].copy() #np.ones((self.size, self.size, self.channels)).astype(np.float32) * self.bg_color
#         label, size, pos = self.add_target(img, num_objects, tex_indices)

#         self.add_noise(img)
#         return img, label, tex_indices, size, pos

#     def __len__(self):
#         return self.image_indices[1] - self.image_indices[0]

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         self.rng.seed(idx)  # to make results repeatable
#         image, label, *_ = self.generate_one()
#         if hasattr(self, "transform"):
#             image = self.transform(image)
#         sample = {'image': image, 'label': label}
#         return sample

#     def implicit_normalization(self, inpt):
#         # since TextureDatasetGenerator.generate_one returns np.uint8 (since it copies from a loaded image), transforms.ToTensor()
#         # will implicitly do a divide by 255. Since we work in [0, 255] space for basically everything, this function is provided
#         # for convenience and should be called just before we pass any tensor into the model. (This means that networks using this
#         # dataset are working in [0, 1] space). The one exception to this is when
#         # you have already called utils.tensorize on the image, which replicates the behaviour of transforms.ToTensor() (but with
#         # better handling of adding dimensions/transposing) and thus will also implicitly do the divide by 255 operation (if the input
#         # type is np.uint8). In summary, always call either utils.tensorize xor this function before passing into the model.
#         return inpt/255.
