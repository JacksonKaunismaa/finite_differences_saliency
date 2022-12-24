import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ColorDatasetGenerator(Dataset):
    def __init__(self, **kwargs):
        # define default values for parameters, can override any using kwargs
        # noise parameters
        self.num_noise = (50, 70)  # number of locations to generate noise at
        self.noise_size = (5, 10)  # size that each noise instance can be

        # image parameters
        self.size = 256 # shape of image
        self.channels = 1  # default is greyscale

        # target parameters
        self.color_classifier = None  # function that maps colors to classes (supports iterables)
        self.num_classes = 2  # how many possible classes there are
        self.color_range = (50, 200)  # range of values that the color-to-be-classified can be
        self.radius = (self.size//6, self.size//3)  # range of possible radii for circles
        self.num_objects = 1 # for multiclass problems, if we want to classify multiple things

        # actual dataset
        #self.labels = {}  # {filename: label} mapping
        self.num_images = 10   # totally arbitrary
        self.options = []  # list of dataset options that need to be saved
        # self.save_dir = ""   # location for all images to be stored
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.options.append(k)

    def add_target(self, arr):
        num_objects = self.num_objects # np.random.randint(*self.num_objects)
        color = np.random.randint(*self.color_range, (num_objects, self.channels)) # rgb for now

        label = np.zeros((self.num_classes))
        label[self.color_classifier(color)] = 1  # multi-hot encoded

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
        return label#, img_name

    def add_noise(self, arr):
        num_noise = np.random.randint(*self.num_noise)
        sizes = np.random.randint(*self.noise_size, num_noise)
        colors = np.random.randint(1, 255, (num_noise, self.channels))
        locations = np.random.randint(self.noise_size[1], self.size-self.noise_size[1], (num_noise, 2))
        for size, color, location in zip(sizes, colors, locations):
            arr[location[0]:location[0]+size,location[1]:location[1]+size] = color

    def generate_one(self, profile=False):
        #benchmark(profile=profile)
        img = np.zeros((self.size, self.size, self.channels))
        #benchmark("initialization", profile)
        label = self.add_target(img)
        #benchmark("circle", profile)
        self.add_noise(img)
        #benchmark("noise", profile)
        return img, label

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
        return self.num_images

    def __getitem__(self, idx):
        print(idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        np.random.seed(idx)  # to make results repeatable
        image, label = self.generate_one()
        sample = {'image': image, 'label': label, "idx": idx}
        if hasattr(self, "transform"):
            sample = self.transform(sample)
        return sample

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

