from source.datasets.base_dataset import BaseDataset
from torchvision.transforms import Resize
from math import ceil, sqrt
from random import shuffle
from glob import glob
from PIL import Image
import numpy as np
import contextlib
import torch
import os


@contextlib.contextmanager
def temp_seed(seed):
    """
    Creates a context in which the seed can be set temporarily.
    :param seed: seed used in every random operations for comparison/reproducibility purpose.
    :return: None.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class BracsBaseDataset(BaseDataset):
    def __init__(self, seed, phase_dict, queries, phase, class_dict, patch_dim, thumbnail_resolution, k,
                 needs_high_resolution=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        Initializes the dataset class which relies on the assumption that the data is stored in a single root
        directory organized in sub-directories (one per class).
        :param seed: seed used in every random operations for comparison/reproducibility purpose.
        :param phase_dict: dictionary containing the proportion of the dataset used by each phase. First key is the
        phase and the second is either 'start' or 'stop'.
        :param queries: a set of queries indicating the location of the images.
        :param phase: phase in which the set is use ('train'/'valid'/'test').
        :param class_dict: dictionary mapping sub-directories name to class indexes (e.g. 'LYM' -> 3).
        :param patch_dim: dimension of the patches.
        :param thumbnail_resolution: number of pixel per patch in the thumbnail.
        :param k: number of patches to be extracted from the images.
        """
        super(BracsBaseDataset, self).__init__(phase, class_dict, patch_dim, thumbnail_resolution, k, mean, std)

        # Get the list of samples for current phase (train / valid / test)
        self.filepaths = self.select_filepaths(seed, phase_dict, queries)

    def select_filepaths(self, seed, phase_dict, queries):
        """
        Randomly selects the filenames for the specified phase ('train'/'valid'/'test').
        :param seed: seed used in every random operations for comparison/reproducibility purpose.
        :param phase_dict: dictionary containing the proportion of the dataset used by each phase. First key is the
        phase and the second is either 'start' or 'stop'.
        :param queries: a set of queries indicating the location of the images.
        :return: array of filenames.
        """
        # Get the name of all files inside the root directory
        filepaths = [filepath for query in queries for filepath in glob(query)]

        # Initialize the selected filepaths list
        selected_filepaths = []

        # Split the filepaths class-wise
        filepaths_dict = {v: [] for v in self.class_dict.values()}
        for k, v in self.class_dict.items():
            filepaths_dict[v].extend(list(filter(lambda filepath: filepath.split(os.sep)[-2] == k, filepaths)))

        # Locally set the seed
        with temp_seed(seed):
            # Iterate over each class and select a sub-sample of all available files
            for k, v in filepaths_dict.items():
                N = len(v)
                phase_start = int(phase_dict[self.phase]['start'] * N)
                phase_stop = int(phase_dict[self.phase]['stop'] * N)
                indexes = np.random.choice(N, replace=False, size=N)
                selected_filepaths.extend(np.array(v)[indexes[phase_start: phase_stop]])
        shuffle(selected_filepaths)
        return selected_filepaths

    def patch_count(self, image):
        # h, w = thumbnail.size
        c, h, w = image.shape
        patch_count = h * w / self.patch_dim ** 2
        return patch_count >= self.k

    def match_minimum_dim(self, image, patch_size):
        # h, w = image.size
        c, h, w = image.shape
        ratio = self.k * patch_size ** 2 / h / w
        h_new = ceil(sqrt(ratio) * h / patch_size) * patch_size
        w_min = ceil(self.k * patch_size / h_new) * patch_size
        w_new = max(w, w_min)

        # Resize the image
        image = Resize(size=(h_new, w_new))(image)
        return image

    def crop_image(self, image):
        c, h, w = image.shape
        h_new = h - (h % self.patch_dim)
        w_new = w - (w % self.patch_dim)
        image = image[:, :h_new, :w_new]
        return image

    def __getitem__(self, index):
        """
        Loads a single pair (input, target) data.
        :param index: index of the sample to load.
        :return: queried pair of (input, target) data.
        """
        # Load the queried image and thumbnail
        filepath = self.filepaths[index]
        image = Image.open(filepath)

        # Normalize the input
        image = self.transform(image)

        # Ensure that the image and thumbnail are large enough
        if not self.patch_count(image):
            image = self.match_minimum_dim(image, self.patch_dim)

        # Retrieve the label
        key = filepath.split('/')[-2].split('.')[0]
        label = torch.tensor(self.class_dict[key])
        return image, label
