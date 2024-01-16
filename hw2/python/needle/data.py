import numpy as np
from .autograd import Tensor, TensorTuple

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any

# following imports are used for loading MNIST data
import struct
import gzip


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        return img[:, ::-1, :] if flip_img else img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        npad = ((self.padding, self.padding), (self.padding, self.padding), (0, 0))
        delta_x = self.padding + delta_x
        delta_y = self.padding + delta_y
        img_pad = np.pad(img, npad)
        return img_pad[delta_x: delta_x + img.shape[0], delta_y: delta_y + img.shape[0], :]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))
    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self._cur_idx = 0
        if self.shuffle:
          order = np.arange(len(self.dataset))
          np.random.shuffle(order)
          self.ordering = np.array_split(order, range(self.batch_size, len(self.dataset), self.batch_size))
        return self
        ### END YOUR SOLUTION

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self._cur_idx < len(self.ordering):
          tuple_arr = self.dataset[self.ordering[self._cur_idx]]
          self._cur_idx += 1
          return tuple(Tensor(a) for a in tuple_arr)
        else:
          raise StopIteration
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms if transforms else []

        with gzip.open(image_filename, 'rb') as f:
          image_content = f.read()
          num_images, num_rows, num_cols = struct.unpack_from('>iii', image_content, 4)
          self.imgs = np.frombuffer(image_content, dtype=np.dtype(np.ubyte).newbyteorder('>'), offset=16).astype(np.float32).reshape(num_images, num_cols, num_rows, 1) / 255
    
        with gzip.open(label_filename, 'rb') as f:
          label_content = f.read()
          self.labels = np.frombuffer(label_content, dtype=np.uint8, offset=8)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        #print('dataset.__getitem__: ',index)
        img = self.imgs[index]
        for tr in self.transforms:
          img = tr(img)
        return img, self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.labels.size
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
