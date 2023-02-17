import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import colorsys
import imageio
import scipy.ndimage
import os
from tqdm import tqdm

class ColorDatasetGenerator(Dataset):
    def __init__(self, **kwargs):
        # define default values for parameters, can override any using kwargs
        # noise parameters
        self.num_noise = (50, 70)  # number of locations to generate noise at
        self.noise_size = (5, 10)  # size that each noise instance can be

        # image parameters
        self.size = 256 # shape of image
        self.channels = 1  # default is greyscale
        self.bg_color = 0  # must have same number of channels 

        # target parameters
        self.color_classifier = None  # function that maps colors to classes (supports iterables)
        self.num_classes = 2  # how many possible classes there are
        # greyscale
        self.color_range = (50, 200)  # range of values that the greyscale color-to-be-classified can be
        # RGB (which we generate as HSV for simplicity)
        self.value_range = (20, 100) # range for value in HSV (subset of (0, 100))
        self.saturation_range = (20, 100) # range for saturation in HSV (subset of (0, 100))
        self.hue_range = (0, 360)  # range for hue in HSV (subset of (0, 360))
    
        self.radius = (self.size//6, self.size//3)  # range of possible radii for circles
        self.num_objects = 1 # supports ranges, if want multiclass

        # actual dataset
        #self.labels = {}  # {filename: label} mapping
        self.image_indices = (0, 1_000_00)   # range of np.random.seeds for the given dataset
        self.options = []  # list of dataset options that need to be saved
        # self.save_dir = ""   # location for all images to be stored
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.options.append(k)
           
    def iterative_color_cvt(self, conversion, iterable):
        # iterates over first dimension
        convert_func = getattr(colorsys, conversion)
        return np.array(list(map(lambda x: convert_func(*x), iterable)))

    def generate_colors(self, amt):
        if self.channels == 1: # greyscale
            color = np.random.randint(*self.color_range, (amt))
        else:  # rgb
            hue = np.random.randint(*self.hue_range, (amt))/360
            saturation = np.random.randint(*self.saturation_range, (amt))/100
            value = np.random.randint(*self.value_range, (amt))/100
            color = (self.iterative_color_cvt("hsv_to_rgb", zip(hue, saturation, value))*255.).round() #(np.array(list(map(lambda x: colorsys.hsv_to_rgb(*x), zip(hue, saturation, value))))*255).round()
        return color

    def add_target(self, arr, set_color):
        num_objects = self.num_objects # np.random.randint(*self.num_objects)
        
        if set_color is not None:
            color = np.array([set_color]*num_objects)
        else:
            color = self.generate_colors(num_objects)

        label = np.zeros((self.num_classes))
        label[self.color_classifier(color)] = 1  # multi-hot encoded
        if self.num_classes == 2:
            label = np.expand_dims(label[0], 0)

        # probably should make sure they dont overlap too much, but num_objects=1 for now
        #img_name = "img"
        radii = np.random.randint(*self.radius, (num_objects))
        locations = np.random.randint(self.radius[1], self.size-self.radius[1], (num_objects, 2))
        for radius, location in zip(radii, locations):
            x_coords = np.arange(radius)
            for x in x_coords:
                height = 2*int(np.sqrt(radius**2 - x**2))
                y_coords = np.arange(height) - height//2 + location[1]
                arr[location[0]+x, y_coords] = color
                arr[location[0]-x, y_coords] = color
            #img_name += f"-{radius}_{location[0]}_{location[1]}"
        #img_name += "".join(random.choice(string.ascii_letters) for _ in range(10)) + ".jpg"
        return label, color, radii, locations  # doesnt really work if we are doing multiclass

    def add_noise(self, arr):
        num_noise = np.random.randint(*self.num_noise)
        sizes = np.random.randint(*self.noise_size, num_noise)
        colors = self.generate_colors(num_noise)
        locations = np.random.randint(self.noise_size[1], self.size-self.noise_size[1], (num_noise, 2))
        for size, color, location in zip(sizes, colors, locations):
            arr[location[0]:location[0]+size,location[1]:location[1]+size] = color

    def generate_one(self, set_color=None, profile=False):
        #benchmark(profile=profile)
        img = np.ones((self.size, self.size, self.channels)) * self.bg_color
        #benchmark("initialization", profile)
        label, color, size, pos = self.add_target(img, set_color)
        #benchmark("circle", profile)
        self.add_noise(img)
        #benchmark("noise", profile)
        return img, label, color, size, pos

    # as it turns out, generating data on the fly is faster any way :\
#     def generate_data(self, amount):
#         for _ in tqdm(range(amount)):
#             img, img_name, label = self.generate_one()
#             pil_img = Image.fromarray(np.squeeze(img.astype(np.uint8)), "L")
#             pil_img.save(os.path.join(self.save_dir, img_name))
#             self.labels[img_name] = label
#         # so we dont have to write a custom Sampler
#         self.data = list(self.labels.items())  # (img_name,label) pairs

    def __len__(self):
        return self.image_indices[1] - self.image_indices[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        np.random.seed(idx)  # to make results repeatable
        image, label, color, _, __ = self.generate_one()
        if hasattr(self, "transform"):
            image = self.transform(image)
        sample = {'image': image, 'label': label, 'color': color, "seeds": idx}
        return sample

    def implicit_normalization(self, inpt):
        # since ColorDatasetGenerator.generate_one returns np.float32, transforms.ToTensor() will implicitly assume we are already 
        #in [0,1] (even though we aren't), and won't do a divide by 255. Since we work in [0, 255] space for basically everything, and
        # the ColorDatasetGenerator models take input in [0, 255] (due to ToTensor having assumed we were in [0,1]), we don't have to 
        # do anything here, and only exists because TextureDatasetGenerator actually requires us to do something here
        return inpt

        # sorta pointless now?
#     def save_options(self, path):
#         save_kwargs = {}
#         for opt in self.options:
#             save_kwargs[opt] = getattr(self, opt)
#         with open(path, "wb") as p:
#             pickle.dump(save_kwargs, p)

#     def load_options(self, path):
#         with open(path, "rb") as p:
#             load_kwargs = pickle.load(p)
#         for k,v in load_kwargs.items():
#             setattr(self, k, v)



class TextureDatasetGenerator(Dataset):
    def __init__(self, dtd_loc, **kwargs):
        super().__init__()
        # define default values for parameters, can override any using kwargs
        # noise parameters
        self.num_noise = (50, 70)  # number of locations to generate noise at
        self.noise_size = (5, 10)  # size that each noise instance can be
#         self.value_range = (20, 100) # range for value in HSV (subset of (0, 100))
#         self.saturation_range = (20, 100) # range for saturation in HSV (subset of (0, 100))
#         self.hue_range = (0, 360)  # range for hue in HSV (subset of (0, 360))

        # image parameters
        self.size = 128 # shape of image
        self.channels = 3  # default is greyscale
        self.bg_color = 0  # must have same number of channels

        # target parameters
        self.num_classes = 2  # how many possible classes there are
        self.textures = []  # list of texture images (order matters)
        self.texture_labels = []  # list of class idxes that each texture is in
        self.texture_file_names = []  # list of filename associated to each texture
        self.texname_to_idx = {}  # maps texture types to their class indices
        self.idx_to_texname = {}  # maps class indices to texture types

        self.radius_frac = (1./6, 1./3)  # range of possible radii (fraction of self.size)
        self.num_objects = 1 # supports ranges, if want multiclass

        # actual dataset
        #self.labels = {}  # {filename: label} mapping
        self.image_indices = (0, 1_000_00)   # range of np.random.seeds for the given dataset
        self.options = []  # list of dataset options that need to be saved
        # self.save_dir = ""   # location for all images to be stored
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.options.append(k)

        if isinstance(dtd_loc, str):
            self.load_dtd_textures(os.path.join(dtd_loc, "images"),
                                os.path.join(dtd_loc, "labels", "labels_joint_anno.txt"))
        elif isinstance(dtd_loc, TextureDatasetGenerator):
            for attrib in ["textures", "texture_file_names", "texture_labels", "num_classes", "idx_to_texname"]:
                setattr(self, attrib, getattr(dtd_loc, attrib))

    def load_dtd_textures(self, images_path, labels_file):
        with open(labels_file, "r") as f:
            labels = f.readlines()
        for label in tqdm(labels):
            name, *categ = label.split()
            if len(categ) > 1: # some textures are multi-class, ignore these for now
                continue
            if categ[0] not in self.texname_to_idx:
                self.texname_to_idx[categ[0]] = len(self.texname_to_idx)
            imread = imageio.v2.imread(os.path.join(images_path, name))

            downsampled = scipy.ndimage.zoom(imread,
                                              [self.size/imread.shape[0], self.size/imread.shape[1], 1.],
                                              order=1)
            self.textures.append(downsampled)
            self.texture_file_names.append(name)
            self.texture_labels.append(self.texname_to_idx[categ[0]])

        self.idx_to_texname = {y:x for x,y in self.texname_to_idx.items()}
        # self.textures = np.asarray(self.textures) # images have different shapes
        self.texture_labels = np.asarray(self.texture_labels)
        self.num_classes = len(self.texname_to_idx)

    @property
    def radius(self):
        return int(self.radius_frac[0]*self.size), int(self.radius_frac[1]*self.size)

    def add_target(self, arr, num_objects, textures):

        # pick a random texture image
        # pick a random location to sample that image at (since our image size is smaller than the texture image size)
        #sample_locs_x = np.random.randint(self.textures[textures[0]].shape[0]-self.size, size=num_objects)
        #sample_locs_y = np.random.randint(self.textures[textures[0]].shape[1]-self.size, size=num_objects)
        label = np.zeros((self.num_classes))
        label[self.texture_labels[textures[0]]] = 1  # multi-hot encoded
        if self.num_classes == 2:
            label = np.expand_dims(label[0], 0)

        # probably should make sure they dont overlap too much, but num_objects=1 for now
        radii = np.random.randint(*self.radius, (num_objects))
        locations = np.random.randint(self.radius[1], self.size-self.radius[1], (num_objects, 2))
        target_zip = zip(radii, locations, textures)#, sample_locs_x, sample_locs_y)
        for radius, location, texture in target_zip:#, sample_loc_x, sample_loc_y in target_zip:
            x_coords = np.arange(radius)
            # subsampling of image guaranteed to be at least size x size
            tex_image = self.textures[texture]#[sample_loc_x:, sample_loc_y:, :]
            for x in x_coords:
                height = 2*int(np.sqrt(radius**2 - x**2))
                y_coords = np.arange(height) - height//2 + location[1]
                arr[location[0]+x, y_coords] = tex_image[location[0]+x, y_coords]
                arr[location[0]-x, y_coords] = tex_image[location[0]-x, y_coords]
        return label, radii, locations  # doesnt really work if we are doing multiclass

    def generate_colors(self, amt):
        return np.random.randint(0,255, size=(amt,3))

    def add_noise(self, arr):
        num_noise = np.random.randint(*self.num_noise)
        sizes = np.random.randint(*self.noise_size, num_noise)
        colors = self.generate_colors(num_noise)
        locations = np.random.randint(self.noise_size[1], self.size-self.noise_size[1], (num_noise, 2))
        for size, color, location in zip(sizes, colors, locations):
            arr[location[0]:location[0]+size,location[1]:location[1]+size] = color

    def generate_one(self):
        num_objects = self.num_objects # np.random.randint(*self.num_objects)
        tex_indices = np.random.randint(len(self.textures), size=num_objects)
        bg_image = np.random.randint(len(self.textures))
        while self.texture_labels[tex_indices[0]] == self.texture_labels[bg_image]:
            bg_image = np.random.randint(len(self.textures))
        img = self.textures[bg_image].copy() #np.ones((self.size, self.size, self.channels)).astype(np.float32) * self.bg_color
        label, size, pos = self.add_target(img, num_objects, tex_indices)

        self.add_noise(img)
        return img, label, tex_indices, size, pos

    def __len__(self):
        return self.image_indices[1] - self.image_indices[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        np.random.seed(idx)  # to make results repeatable
        image, label, *_ = self.generate_one()
        if hasattr(self, "transform"):
            image = self.transform(image)
        sample = {'image': image, 'label': label}
        return sample

    def implicit_normalization(self, inpt):
        # since TextureDatasetGenerator.generate_one returns np.uint8 (since it copies from a loaded image), transforms.ToTensor()
        # will implicitly do a divide by 255. Since we work in [0, 255] space for basically everything, this function is provided
        # for convenience and should be called just before we pass any tensor into the model. The one exception to this is when
        # you have already called utils.tensorize on the image, which replicates the behaviour of transforms.ToTensor() (but with
        # better handling of adding dimensions/transposing) and thus will also implicitly do the divide by 255 operation (if the input
        # type is np.uint8). In summary, always call either utils.tensorize xor this function before passing into the model.
        return inpt/255.
