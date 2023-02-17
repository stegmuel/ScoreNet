from torchvision.transforms import ToTensor, Resize, Compose, RandomHorizontalFlip, Normalize, RandomApply,\
    RandomGrayscale, ColorJitter
from source.utils import dino_utils
import torch
import abc


class BaseDataset(abc.ABC):
    def __init__(self, phase, class_dict, patch_dim, thumbnail_resolution, k, mean, std):
        """
        Initializes the dataset class which relies on the assumption that the data is stored in a single root
        directory organized in sub-directories (one per class).
        phase and the second is either 'start' or 'stop'.
        :param phase: phase in which the set is use ('train'/'valid'/'test').
        :param class_dict: dictionary mapping sub-directories name to class indexes (e.g. 'LYM' -> 3).
        :param patch_dim: dimension of the patches.
        :param thumbnail_resolution: number of pixel per patch in the thumbnail.
        :param k: number of patches to be extracted from the images.
        """
        super(BaseDataset, self).__init__()

        # Store basic parameters
        self.k = k
        self.mean = mean
        self.std = std
        self.phase = phase
        self.filepaths = []
        self.patch_dim = patch_dim
        self.class_dict = class_dict
        self.thumbnail_resolution = thumbnail_resolution
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Get the normalizer
        self.T = ToTensor()

        # Get the image transformer
        self.transform = self.get_image_transformer()

    def get_image_transformer(self):
        normalize = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])
        if self.phase == 'train':
            flip_and_color_jitter = Compose([
                RandomHorizontalFlip(p=0.5),
                RandomApply([ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
                RandomGrayscale(p=0.2)])
            transform = Compose([
                flip_and_color_jitter,
                dino_utils.GaussianBlur(0.1),
                dino_utils.Solarization(0.2),
                normalize,
            ])
        else:
            transform = normalize
        return transform

    def __len__(self):
        return len(self.filepaths)

    def get_thumbnail(self, large_image):
        """
        Downscale the large image.
        :param large_image: image to downscale.
        :return: downscaled image.
        """
        _, H, W = large_image.shape
        h_patches, w_patches = H // self.patch_dim, W // self.patch_dim
        if h_patches == 0 or w_patches == 0:
            raise RuntimeError
        h, w = h_patches * self.thumbnail_resolution, w_patches * self.thumbnail_resolution
        thumbnail = Resize(size=(h, w))(large_image)
        return thumbnail

    @abc.abstractmethod
    def select_filepaths(self, **args):
        pass

    @abc.abstractmethod
    def __getitem__(self, index):
        pass
