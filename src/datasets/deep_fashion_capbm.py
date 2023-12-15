import torch
import torchvision

from torch.utils.data import Dataset
import torchvision.io

import os
import random
import itertools

import numpy as np

import src.utils.memory
import src.utils.pkl



class CatAndAttrPredBmkDataset(Dataset):
    """
    Dataset class for the DeepFashion Category and Attribute Prediction Benchmark.
    """

    def __init__(
            self,
            dataset_dirname,
            img_transform = None
            ):
        """
        :param dataset_dirname: str
            Directory where the dataset is stored.
        :param img_transform: torch.Tensor -> torch.Tensor, default=None
            Transformation to apply to images.
        """

        if not os.path.exists(dataset_dirname):
            raise ValueError("Directory {:s} does not exist".format(dataset_dirname))

        self._dataset_dirname = dataset_dirname
        self._img_transform = img_transform

        self._compute_img_subdir_vars()
        self._compute_num_imgs()

        self._compute_masks()

        self._store_img_filename_codes()
        self._store_img_bbox_codes()
        self._store_img_cat_codes()


    def __len__(self):
        
        return self._num_imgs    


    def __getitem__(self, idx):
        """
        Returns a dataset item.

        :param idx: int
            Index of the dataset item.

        :return img: torch.tensor
            An image of the dataset.
        :return cat_id: int
            Category ID of the image.
        """

        img_filename = self._decode_img_filename(self._get_img_filename_code(idx))
        img_cat = self._decode_img_cat(self._get_img_cat_code(idx))

        full_img_filename = os.path.join(self._dataset_dirname, img_filename)
        img = torchvision.io.read_image(full_img_filename)
        if img.size(dim=0) == 1: img = img.repeat(3, 1, 1)

        img_bbox = self._decode_img_bbox(self._get_img_bbox_code(idx))

        x1, y1, x2, y2 = img_bbox

        img = img[:,y1:y2,x1:x2]

        if self._img_transform is not None:
            img = self._img_transform(img)

        return img, img_cat


    def get_subset_indices(self, split=None):
        """
        Returns dataset subset indices.

        :param split: str, optional
            Split name. Can be "train", "test", or "val".
            Split will not be taken into account if not provided.
        
        :return: np.ndarray
            An array with the split indices.
        """

        mask = np.full(shape=(self._num_imgs), fill_value=True)

        if split is not None:

            if split == "train": split_num = 0
            elif split == "val": split_num = 1
            elif split == "test": split_num = 2

            mask = np.logical_and(mask, self._split_num_mask == split_num)

        idxs = np.argwhere(mask).flatten()

        return idxs


    ########
    # AUXILIARY INITIALIZATION METHODS
    ########


    def _compute_img_subdir_vars(self):
        """
        Auxiliary function called in the constructor.

        Computes image subdirectory names the `img` directory and stores them into attributes
          - `_img_subdir_name_inv_dict`: Dict that returns an idx given an image subdir.
        """

        img_dirname = os.path.join(self._dataset_dirname, "img")
        img_subdir_name_list = [name for name in os.listdir(img_dirname) if name[0] != "."]

        self._img_subdir_name_inv_dict = {
            img_subdir_name: idx for (idx, img_subdir_name) in enumerate(img_subdir_name_list)
        }


    def _compute_num_imgs(self):
        """
        Auxiliary function called in the constructor.
        
        Computes the number of images in the dataset.
        The number of images is stored in attribute `_num_imgs`.
        The number is read from the `Eval/list_eval_partition.txt` file.
        """

        list_eval_partition_filename = os.path.join(self._dataset_dirname, "Eval", "list_eval_partition.txt")
        
        list_eval_partition_file = open(list_eval_partition_filename, "r")

        self._num_imgs = int(list_eval_partition_file.readline()[:-1])

        list_eval_partition_file.close()


    def _compute_masks(self):
        """
        Auxiliary function called in the constructor.

        Computes train/test/val split masks.
        Indices are stored in the following attributes:
          - `_split_num_mask`, where train/val/test is 0/1/2
        """

        # Split mask

        self._split_num_mask = np.empty(shape=(self._num_imgs), dtype=int)

        list_eval_partition_filename = os.path.join(self._dataset_dirname, "Eval", "list_eval_partition.txt")
        list_eval_partition_file = open(list_eval_partition_filename, "r")

        for _ in range(2):
            line = list_eval_partition_file.readline()

        for img_idx, line in enumerate(list_eval_partition_file.readlines()):

            split_name = line.split()[1]
            
            if split_name == "train": self._split_num_mask[img_idx] = 0
            elif split_name == "val": self._split_num_mask[img_idx] = 1
            elif split_name == "test": self._split_num_mask[img_idx] = 2

        list_eval_partition_file.close()


    def _store_img_filename_codes(self):
        """
        Auxiliary function called in the constructor.
        
        Encodes image filenames and stores them in the `_img_filename_codes_arr` attribute.
        Image filenames are read from the `Eval/list_eval_partition.txt` file.
        """

        self._img_filename_codes_arr = bytearray(3 * self._num_imgs)

        img_filename_list_filename = os.path.join(self._dataset_dirname, "Eval", "list_eval_partition.txt")
        img_filename_list_file = open(img_filename_list_filename, "r")

        for _ in range(2):
            img_filename_list_file.readline()

        for idx, line in enumerate(img_filename_list_file.readlines()):
            
            img_filename_code = self._encode_img_filename(line.split()[0])          
            self._set_img_filename_code(idx, img_filename_code)

        img_filename_list_file.close()


    def _store_img_bbox_codes(self):
        """
        Auxiliary function called in the constructor.
        
        Encodes image bboxes and stores them in the `_img_bbox_codes_arr` attribute.
        Image bboxes are read from the `Anno_coarse/list_bbox.txt` file.
        """

        self._img_bbox_codes_arr = bytearray(5 * self._num_imgs)

        img_bbox_filename = os.path.join(self._dataset_dirname, "Anno_coarse", "list_bbox.txt")
        img_bbox_file = open(img_bbox_filename, "r")

        for _ in range(2):
            img_bbox_file.readline()

        for idx, line in enumerate(img_bbox_file.readlines()):
            
            tkns = line.split()
            img_bbox = [int(tkn) for tkn in tkns[-4:]]
            img_bbox_code = self._encode_img_bbox(img_bbox)
            self._set_img_bbox_code(idx, img_bbox_code)

        img_bbox_file.close()


    def _store_img_cat_codes(self):
        """
        Auxiliary function called in the constructor.
        
        Encodes image categories and stores them in the `_img_cat_codes_arr` attribute.
        Image categories are read from the `Anno_coarse/list_category_img.txt` file.
        """

        self._img_cat_codes_arr = bytearray(self._num_imgs)

        img_cat_filename = os.path.join(self._dataset_dirname, "Anno_coarse", "list_category_img.txt")
        img_cat_file = open(img_cat_filename, "r")

        for _ in range(2):
            img_cat_file.readline()

        for idx, line in enumerate(img_cat_file.readlines()):
            
            img_cat = int(line.split()[-1])
            img_cat_code = self._encode_img_cat(img_cat)
            self._set_img_cat_code(idx, img_cat_code)

        img_cat_file.close()


    ########
    # AUXILIARY CODE AND READ METHODS
    ########


    ####
    # IMAGE FILENAME
    ####


    def _encode_img_filename(self, img_filename):

        img_filename_tkns = img_filename.split(os.path.sep)
        img_filename_code = bytearray(3)

        # Image subdir name

        img_subdir_name = img_filename_tkns[1]
        img_subdir_zid = self._img_subdir_name_inv_dict[img_subdir_name]

        img_filename_code[0] = img_subdir_zid >> 8
        img_filename_code[1] = img_subdir_zid & 0b11111111

        # Image ID

        img_name = img_filename_tkns[2]
        img_id = int(img_name[4:-4])

        img_filename_code[2] = img_id

        return img_filename_code
    

    def _decode_img_filename(self, img_filename_code):

        # Image subdir name

        img_subdir_zid = img_filename_code[0] << 8 | img_filename_code[1]

        img_subdir_name = next(itertools.islice(self._img_subdir_name_inv_dict.keys(), img_subdir_zid, None))

        # Domain type and image ID

        img_id = img_filename_code[2]

        img_name = "img_{:08d}.jpg".format(img_id)

        # Build image filename

        img_filename = os.path.join("img", img_subdir_name, img_name)

        return img_filename
    

    def _set_img_filename_code(self, idx, img_filename_code):

        self._img_filename_codes_arr[3*idx:3*(idx+1)] = img_filename_code[:]
    

    def _get_img_filename_code(self, idx):

        return self._img_filename_codes_arr[3*idx:3*(idx+1)]
    

    ####
    # IMAGE BBOX
    ####


    def _encode_img_bbox(self, img_bbox):

        img_bbox_code = bytearray(5)

        img_bbox_code[4] = 0
        for idx in range(4):
            img_bbox_code[idx] = img_bbox[idx] & 0b11111111
            img_bbox_code[4] |= (img_bbox[idx] & 0b100000000) >> (5 + idx)

        return img_bbox_code


    def _decode_img_bbox(self, img_bbox_code):

        img_bbox = [0 for _ in range(4)]

        for idx in range(4):
            img_bbox[idx] = img_bbox_code[idx] + ((img_bbox_code[4] << (5 + idx)) & 0b100000000)
            
        return img_bbox


    def _set_img_bbox_code(self, idx, img_bbox_code):

        self._img_bbox_codes_arr[5*idx:5*(idx+1)] = img_bbox_code[:]
    

    def _get_img_bbox_code(self, idx):

        return self._img_bbox_codes_arr[5*idx:5*(idx+1)]
    

    ####
    # IMAGE CATEGORY
    ####


    def _encode_img_cat(self, img_cat):

        img_cat_code = bytearray(1)
        img_cat_code[0] = img_cat

        return img_cat_code


    def _decode_img_cat(self, img_cat_code):

        return img_cat_code[0]


    def _set_img_cat_code(self, idx, img_cat_code):

        self._img_cat_codes_arr[idx:idx+1] = img_cat_code[:]
    

    def _get_img_cat_code(self, idx):

        return self._img_cat_codes_arr[idx:idx+1]


    ########
    # DEBUG METHODS
    ########


    def _num_bytes(self):
        """
        Computes the memory usage to store all attributes of this object.
        Used only for debug purposes.

        :return num_bytes: int 
            Memory usage in bytes.
        """

        num_bytes = 0

        num_bytes += src.utils.memory.get_num_bytes(self._dataset_dirname)
        num_bytes += src.utils.memory.get_num_bytes(self._img_transform)

        num_bytes += src.utils.memory.get_num_bytes(self._split_num_mask)

        num_bytes += src.utils.memory.get_num_bytes(self._img_subdir_name_inv_dict)

        num_bytes += src.utils.memory.get_num_bytes(self._num_imgs)

        num_bytes += src.utils.memory.get_num_bytes(self._img_filename_codes_arr)
        num_bytes += src.utils.memory.get_num_bytes(self._img_bbox_codes_arr)
        num_bytes += src.utils.memory.get_num_bytes(self._img_cat_codes_arr)

        return num_bytes


    def _is_pickable(self):
        """
        Attempts to pickle all attributes of this class and prints the respective status.
        Used only for debug purposes.
        """

        print("-- PICKLE STATUS START --")

        print("self._dataset_dirname")
        print("  ", src.utils.pkl.pickle_test(self._dataset_dirname))
        print("self._img_transform")
        print("  ", src.utils.pkl.pickle_test(self._img_transform))

        print("self._split_num_mask")
        print("  ", src.utils.pkl.pickle_test(self._split_num_mask))

        print("self._img_subdir_name_inv_dict")
        print("  ", src.utils.pkl.pickle_test(self._img_subdir_name_inv_dict))
        
        print("self._num_imgs")
        print("  ", src.utils.pkl.pickle_test(self._num_imgs))
        
        print("self._img_filename_codes_arr")
        print("  ", src.utils.pkl.pickle_test(self._img_filename_codes_arr))
        print("self._img_bbox_codes_arr")
        print("  ", src.utils.pkl.pickle_test(self._img_bbox_codes_arr))
        print("self._img_cat_codes_arr")
        print("  ", src.utils.pkl.pickle_test(self._img_cat_codes_arr))

        print("-- PICKLE STATUS END --")
