import torch
import torchvision

from torch.utils.data import Dataset
import torchvision.io

import os
from multiprocessing import Lock

import random

import utils.mem
import utils.pkl

import numpy as np



class CatAttrPredBM(Dataset):
    """
    PyTorch Dataset class.
    Dataset: DeepFashion Category and Attribute Prediction Benchmark.

    Currently supported data:
      - Image cropped with bbox.
      - Category index (50 categories):
        - Value in the [0, 49] range.
      - Attribute annotations (1000 attributes):
        - Neg/Unk/Pos: -1/0/1

    Images from the dataset come in different sizes. To use with PyTorch DataLoader, pass a
    resizing transform (i.e. torchvision.transforms.Resize) in the constructor.

    This dataset already provides train/test/val data splits. Call ``get_split_mask_idxs()``, 
    to get the respective indices.

    Category/supcategory names can be checked by calling ``get_category_name()``.
    Attribute/supattribute names can be checked by calling ``get_attribute_name()``.

    Supports num_workers > 1 when used with PyTorch DataLoader.
    """

    def __init__(
            self,
            dataset_dir,
            img_transform = None
            ):
        """
        :param dataset_dir: str
            Directory where the dataset is stored.
        :param img_transform: torch.Tensor -> torch.Tensor
            Transformation to apply to images.
        """

        if not os.path.exists(dataset_dir):
            raise ValueError("Directory {:s} does not exist".format(dataset_dir))

        self._dataset_dir = dataset_dir
        self._img_transform = img_transform

        self._load_category_metadata()
        self._load_attribute_metadata()
        self._load_split_masks()
        
        self._compute_list_eval_partition_cursors()
        self._compute_image_bbox_cursors()
        self._compute_image_category_cursors()
        self._compute_image_attribute_cursors()

        self._open_data_files()
        self._initialize_locks()


    def __len__(self):

        return len(self._list_eval_partition_cursor_list)


    def __getitem__(self, idx):

        full_image_filename = os.path.join(self._dataset_dir, self._read_list_eval_partition_el(idx))
        image = torchvision.io.read_image(full_image_filename).double()

        x1, y1, x2, y2 = self._read_image_bbox_el(idx)
        image = image[:,y1:y2,x1:x2]

        if self._img_transform is not None:
            image = self._img_transform(image)

        cat_idx = self._read_image_category_el(idx) - 1

        attr_anno = torch.Tensor(self._read_image_attribute_el(idx))

        return image, cat_idx, attr_anno


    def get_category_name(self, cat_idx):
        """
        Returns the name of a category and its corresponding supcategory.

        :param cat_idx: int
            The category index. Must be an integer in the [0, 49] range.
        
        :return cat_name: str
            The category name.
        :return supcat_name: str
            The supcategory name.
        """

        return (
            self._category_name_list[cat_idx],
            self._supcategory_name_list[self._supcategory_idx_list[cat_idx]]
        )


    def get_attribute_name(self, attr_idx):
        """
        Returns the name of an attribute and its corresponding supattribute.

        :param attr_idx: int
            The attribute index. Must be an integer in the [0, 999] range.
        
        :return cat_name: str
            The attribute name.
        :return supcat_name: str
            The supattribute name.
        """

        return (
            self._attribute_name_list[attr_idx],
            self._supattribute_name_list[self._supattribute_idx_list[attr_idx]]
        )


    def get_split_mask_idxs(self, split_name):
        """
        Returns train/test/val split indices.

        :param split_name: str
            Split name. Can be "train", "test", or "val".
        
        :return: list
            A list with the split indices.
        """

        if split_name == "train": return self._train_mask_idxs
        elif split_name == "test": return self._test_mask_idxs
        elif split_name == "val": return self._val_mask_idxs


    def _load_category_metadata(self):
        """
        Reads and defines category and supcategory names.
        Auxiliary function called in the constructor.
        """

        # Read category names and supcategory idxs        

        self._category_name_list = []
        self._supcategory_idx_list = []

        category_name_filename = os.path.join(self._dataset_dir, "Anno_coarse", "list_category_cloth.txt")
        category_name_file = open(category_name_filename, 'r')

        for _ in range(2):
            line = category_name_file.readline()

        for line in category_name_file.readlines():
            tkns = line.split()
            self._category_name_list.append(tkns[0])
            self._supcategory_idx_list.append(int(tkns[1]))

        category_name_file.close()

        # Define supcategory names

        self._supcategory_name_list = [
            "Upper-body",
            "Lower-body",
            "Full-body"
        ]


    def _load_attribute_metadata(self):
        """
        Reads and defines attribute and supattribute names.
        Auxiliary function called in the constructor.
        """

        # Read attribute names and supcategory idxs

        self._attribute_name_list = []
        self._supattribute_idx_list = []

        attribute_name_filename = os.path.join(self._dataset_dir, "Anno_coarse", "list_attr_cloth.txt")
        attribute_name_file = open(attribute_name_filename, 'r')        

        for _ in range(2):
            line = attribute_name_file.readline()

        for line in attribute_name_file.readlines():
            tkns = line.split()
            self._attribute_name_list.append(" ".join(tkns[:-1]))
            self._supattribute_idx_list.append(int(tkns[-1]))

        attribute_name_file.close()

        # Define supattribute names

        self._supattribute_name_list = [
            "Texture",
            "Fabric",
            "Shape",
            "Part",
            "Style"
        ]


    def _load_split_masks(self):
        """
        Reads train/test/val split indices.
        Auxiliary function called in the constructor.
        """
        
        self._train_mask_idxs = []
        self._test_mask_idxs = []
        self._val_mask_idxs = []

        split_mask_filename = os.path.join(self._dataset_dir, "Eval", "list_eval_partition.txt")
        split_mask_file = open(split_mask_filename, 'r')

        for _ in range(2):
            line = split_mask_file.readline()

        for img_idx, line in enumerate(split_mask_file.readlines()):
            split_name = line.split()[1]
            if split_name == "train": self._train_mask_idxs.append(img_idx)
            elif split_name == "test": self._test_mask_idxs.append(img_idx)
            elif split_name == "val": self._val_mask_idxs.append(img_idx)

        split_mask_file.close()


    def _compute_list_eval_partition_cursors(self):
        """
        Computes file cursor values from the train/test/val split file.
        Auxiliary function called in the constructor.
        """

        self._list_eval_partition_cursor_list = []

        list_eval_partition_filename = os.path.join(self._dataset_dir, "Eval", "list_eval_partition.txt")
        list_eval_partition_file = open(list_eval_partition_filename, 'r')

        for _ in range(2):
            line = list_eval_partition_file.readline()

        cursor = list_eval_partition_file.tell()
        line = list_eval_partition_file.readline()

        while line != "":
            
            self._list_eval_partition_cursor_list.append(cursor)
            cursor = list_eval_partition_file.tell()
            line = list_eval_partition_file.readline()

        list_eval_partition_file.close()


    def _compute_image_bbox_cursors(self):
        """
        Computes file cursor values from the bbox file.
        Auxiliary function called in the constructor.
        """

        self._image_bbox_cursor_list = []

        image_bbox_filename = os.path.join(self._dataset_dir, "Anno_coarse", "list_bbox.txt")
        image_bbox_file = open(image_bbox_filename, 'r')

        for _ in range(2):
            line = image_bbox_file.readline()

        cursor = image_bbox_file.tell()
        line = image_bbox_file.readline()

        while line != "":

            self._image_bbox_cursor_list.append(cursor)
            cursor = image_bbox_file.tell()
            line = image_bbox_file.readline()

        image_bbox_file.close()


    def _compute_image_category_cursors(self):
        """
        Computes file cursor values from the category file.
        Auxiliary function called in the constructor.
        """

        self._image_category_cursor_list = []

        image_category_filename = os.path.join(self._dataset_dir, "Anno_coarse", "list_category_img.txt")
        image_category_file = open(image_category_filename, 'r')

        for _ in range(2):
            line = image_category_file.readline()

        cursor = image_category_file.tell()
        line = image_category_file.readline()

        while line != "":

            self._image_category_cursor_list.append(cursor)
            cursor = image_category_file.tell()
            line = image_category_file.readline()

        image_category_file.close()


    def _compute_image_attribute_cursors(self):
        """
        Computes file cursor values from the attribute file.
        Auxiliary function called in the constructor.
        """

        self._image_attribute_cursor_list = []

        image_attribute_filename = os.path.join(self._dataset_dir, "Anno_coarse", "list_attr_img.txt")
        image_attribute_file = open(image_attribute_filename, 'r')

        for _ in range(2):
            line = image_attribute_file.readline()

        cursor = image_attribute_file.tell()
        line = image_attribute_file.readline()

        while line != "":

            self._image_attribute_cursor_list.append(cursor)
            cursor = image_attribute_file.tell()
            line = image_attribute_file.readline()

        image_attribute_file.close()


    def _open_data_files(self):
        """
        Opens the required data files for reading.
        Auxiliary function called in the constructor.
        """

        list_eval_partition_filename = os.path.join(self._dataset_dir, "Eval", "list_eval_partition.txt")
        self._list_eval_partition_file = open(list_eval_partition_filename, 'r')

        image_bbox_filename = os.path.join(self._dataset_dir, "Anno_coarse", "list_bbox.txt")
        self._image_bbox_file = open(image_bbox_filename, 'r')

        image_category_filename = os.path.join(self._dataset_dir, "Anno_coarse", "list_category_img.txt")
        self._image_category_file = open(image_category_filename, 'r')

        image_attribute_filename = os.path.join(self._dataset_dir, "Anno_coarse", "list_attr_img.txt")
        self._image_attribute_file = open(image_attribute_filename, 'r')


    def _initialize_locks(self):
        """
        Initializes multiprocessing locks for data files.
        Auxiliary function called in the constructor.
        """

        self._list_eval_partition_file_lock = Lock()
        self._image_bbox_file_lock = Lock()
        self._image_category_file_lock = Lock()
        self._image_attribute_file_lock = Lock()


    def _read_list_eval_partition_el(self, idx):
        """
        Reads an image filename from the train/test/val split file.
        Auxiliary function called when loading a data point.
        """
        
        with self._list_eval_partition_file_lock:

            self._list_eval_partition_file.seek(self._list_eval_partition_cursor_list[idx])
            line = self._list_eval_partition_file.readline()

        tkns = line.split()

        image_filename = tkns[0]
        
        return image_filename
    

    def _read_image_bbox_el(self, idx):
        """
        Reads an image bbox from the bbox file.
        Auxiliary function called when loading a data point.
        """
        
        with self._image_bbox_file_lock:
        
            self._image_bbox_file.seek(self._image_bbox_cursor_list[idx])
            line = self._image_bbox_file.readline()

        tkns = line.split()

        image_bbox = [int(tkn) for tkn in tkns[1:5]]
        
        return image_bbox


    def _read_image_category_el(self, idx):
        """
        Reads an image filename from the category file.
        Auxiliary function called when loading a data point.

        Can be used to separatedly read image categories from the dataset to compute stats.
        """
        
        with self._image_category_file_lock:

            self._image_category_file.seek(self._image_category_cursor_list[idx])
            line = self._image_category_file.readline()

        tkns = line.split()

        image_category = int(tkns[1])

        return image_category


    def _read_image_attribute_el(self, idx):
        """
        Reads image attributes from the attribute file.
        Auxiliary function called when loading a data point.

        Can be used to separatedly read image attributes from the dataset to compute stats.
        """
        
        with self._image_attribute_file_lock:

            self._image_attribute_file.seek(self._image_attribute_cursor_list[idx])
            line = self._image_attribute_file.readline()

        tkns = line.split()

        image_attributes = [int(tkn) for tkn in tkns[1:]]

        return image_attributes


    def _num_bytes(self):
        """
        Computes the memory usage to store all attributes of this object.
        Used only for debug purposes.

        :return num_bytes: int 
            Memory usage in bytes.
        """

        num_bytes = 0

        num_bytes += utils.mem.get_num_bytes(self._dataset_dir)
        num_bytes += utils.mem.get_num_bytes(self._img_transform)

        num_bytes += utils.mem.get_num_bytes(self._train_mask_idxs)
        num_bytes += utils.mem.get_num_bytes(self._test_mask_idxs)
        num_bytes += utils.mem.get_num_bytes(self._val_mask_idxs)

        num_bytes += utils.mem.get_num_bytes(self._category_name_list)
        num_bytes += utils.mem.get_num_bytes(self._supcategory_idx_list)
        num_bytes += utils.mem.get_num_bytes(self._supcategory_name_list)

        num_bytes += utils.mem.get_num_bytes(self._attribute_name_list)
        num_bytes += utils.mem.get_num_bytes(self._supattribute_idx_list)
        num_bytes += utils.mem.get_num_bytes(self._supattribute_name_list)

        num_bytes += utils.mem.get_num_bytes(self._list_eval_partition_cursor_list)
        num_bytes += utils.mem.get_num_bytes(self._image_bbox_cursor_list)
        num_bytes += utils.mem.get_num_bytes(self._image_category_cursor_list)
        num_bytes += utils.mem.get_num_bytes(self._image_attribute_cursor_list)

        num_bytes += utils.mem.get_num_bytes(self._list_eval_partition_file)
        num_bytes += utils.mem.get_num_bytes(self._image_bbox_file)
        num_bytes += utils.mem.get_num_bytes(self._image_category_file)
        num_bytes += utils.mem.get_num_bytes(self._image_attribute_file)

        num_bytes += utils.mem.get_num_bytes(self._list_eval_partition_file_lock)
        num_bytes += utils.mem.get_num_bytes(self._image_bbox_file_lock)
        num_bytes += utils.mem.get_num_bytes(self._image_category_file_lock)
        num_bytes += utils.mem.get_num_bytes(self._image_attribute_file_lock)

        return num_bytes



class ConsToShopClothRetrBM(Dataset):
    """
    PyTorch Dataset class.
    Dataset: DeepFashion Consumer-to-shop Clothes Retrieval Benchmark

    Currently supported data:
      - Image cropped with bbox.
      - Attribute annotations (303 attributes):
        - Neg/Unk/Pos: -1/0/1

    Images from the dataset come in different sizes. To use with PyTorch DataLoader, pass a
    resizing transform (i.e. torchvision.transforms.Resize) in the constructor.

    This dataset already provides train/test/val data splits. Call ``get_split_mask_idxs()``, 
    to get the respective indices.

    Attribute/supattribute names can be checked by calling ``get_attribute_name()``.

    Supports num_workers > 1 when used with PyTorch DataLoader.
    """

    def __init__(
            self,
            dataset_dir,
            img_transform = None,
            neg_aux_filename_id = None
            ):
        """
        :param dataset_dir: str
            Directory where the dataset is stored.
        :param img_transform: torch.Tensor -> torch.Tensor, default=None
            Transformation to apply to images.
        :param neg_aux_filename_id: str, default=None
            ID for the negative image idxs file. If no such file with this ID exists in the
            aux subdirectory of the dataset directory, a new one will be created. 
            If no ID is provided, a non-id file with random negative image idxs will be
            recalculated and used.
        """

        if not os.path.exists(dataset_dir):
            raise ValueError("Directory {:s} does not exist".format(dataset_dir))

        self._dataset_dir = dataset_dir
        self._img_transform = img_transform

        self._load_attribute_metadata()
        self._load_split_masks()
        
        self._list_eval_partition_aux_idxs_filename = os.path.join(self._dataset_dir, "aux", "list_eval_partition_aux_idxs.txt")
        if not os.path.exists(self._list_eval_partition_aux_idxs_filename):
            self._write_list_eval_partition_aux_idxs_file()

        self._list_eval_partition_neg_aux_idxs_filename = "list_eval_partition_neg_aux_idxs"
        if neg_aux_filename_id is not None: self._list_eval_partition_neg_aux_idxs_filename += "__" + neg_aux_filename_id
        self._list_eval_partition_neg_aux_idxs_filename += ".txt"
        self._list_eval_partition_neg_aux_idxs_filename = os.path.join(self._dataset_dir, "aux", self._list_eval_partition_neg_aux_idxs_filename)
        if not os.path.exists(self._list_eval_partition_neg_aux_idxs_filename) or neg_aux_filename_id is None:
            self._write_list_eval_partition_neg_aux_idxs_file()

        self._compute_list_eval_partition_aux_idxs_cursors()
        self._compute_list_eval_partition_neg_aux_idxs_cursors()
        self._compute_image_filename_bbox_cursors()
        self._compute_item_attribute_cursors()

        self._open_data_files()
        self._initialize_locks()


    def __len__(self):

        return len(self._image_filename_bbox_cursor_list)


    def __getitem__(self, idx):

        image_bbox_idx_1, image_bbox_idx_2, item_attr_idx = self._read_list_eval_partition_aux_idxs_el(idx)
        image_bbox_idx_neg, _ = self._read_list_eval_partition_neg_aux_idxs_el(idx)

        image_filename_1, (x1__1, y1__1, x2__1, y2__1) = self._read_image_filename_bbox_el_(image_bbox_idx_1)
        image_filename_2, (x1__2, y1__2, x2__2, y2__2) = self._read_image_filename_bbox_el_(image_bbox_idx_2)
        image_filename_3, (x1__3, y1__3, x2__3, y2__3) = self._read_image_filename_bbox_el_(image_bbox_idx_neg)

        full_image_filename_1 = os.path.join(self._dataset_dir, image_filename_1)
        image_1 = torchvision.io.read_image(full_image_filename_1)
        if image_1.size(dim=0) == 1: image_1 = image_1.repeat(3, 1, 1)

        full_image_filename_2 = os.path.join(self._dataset_dir, image_filename_2)
        image_2 = torchvision.io.read_image(full_image_filename_2)
        if image_2.size(dim=0) == 1: image_2 = image_2.repeat(3, 1, 1)

        full_image_filename_3 = os.path.join(self._dataset_dir, image_filename_3)
        image_3 = torchvision.io.read_image(full_image_filename_3)
        if image_3.size(dim=0) == 1: image_3 = image_3.repeat(3, 1, 1)

        image_1 = image_1[:,y1__1:y2__1,x1__1:x2__1]
        image_2 = image_2[:,y1__2:y2__2,x1__2:x2__2]
        image_3 = image_3[:,y1__3:y2__3,x1__3:x2__3]

        if self._img_transform is not None:
            image_1 = self._img_transform(image_1)
            image_2 = self._img_transform(image_2)
            image_3 = self._img_transform(image_3)

        item_attr_anno = torch.Tensor(self._read_item_attribute_el(item_attr_idx))

        return image_1, image_2, image_3, item_attr_anno


    def get_attribute_name(self, attr_idx):
        """
        Returns the name of an attribute and its corresponding supattribute.

        :param attr_idx: int
            The attribute index. Must be an integer in the [0, 999] range.
        
        :return cat_name: str
            The attribute name.
        :return supcat_name: str
            The supattribute name.
        """

        return (
            self._attribute_name_list[attr_idx],
            self._supattribute_name_list[self._supattribute_idx_list[attr_idx]]
        )


    def get_split_mask_idxs(self, split_name):
        """
        Returns train/test/val split indices.

        :param split_name: str
            Split name. Can be "train", "test", or "val".
        
        :return: list
            A list with the split indices.
        """

        if split_name == "train": return self._train_mask_idxs
        elif split_name == "test": return self._test_mask_idxs
        elif split_name == "val": return self._val_mask_idxs


    def _load_attribute_metadata(self):
        """
        Reads and defines attribute and supattribute names.
        Auxiliary function called in the constructor.
        """

        # Read attribute names and supattribute idxs

        self._attribute_name_list = []
        self._supattribute_idx_list = []

        attribute_name_filename = os.path.join(self._dataset_dir, "Anno", "list_attr_cloth.txt")
        attribute_name_file = open(attribute_name_filename, 'r')

        for _ in range(2):
            line = attribute_name_file.readline()

        for line in attribute_name_file.readlines():

            tkns = line.split()

            if tkns[1][0] == "(": start_tkn_idx = 2
            else: start_tkn_idx = 1

            self._attribute_name_list.append(" ".join(tkns[start_tkn_idx:-1]))
            self._supattribute_idx_list.append(int(tkns[-1]))

        attribute_name_file.close()

        # Read supattribute names

        self._supattribute_name_list = []

        supattribute_name_filename = os.path.join(self._dataset_dir, "Anno", "list_attr_type.txt")
        supattribute_name_file = open(supattribute_name_filename, 'r')    

        for _ in range(2):
            line = supattribute_name_file.readline()

        for line in supattribute_name_file.readlines():

            tkns = line.split()

            if tkns[1][0] == "(": start_tkn_idx = 2
            else: start_tkn_idx = 1

            self._supattribute_name_list.append(" ".join(tkns[start_tkn_idx:-1]))

        supattribute_name_file.close()


    def _load_split_masks(self):
        """
        Reads train/test/val split indices.
        Auxiliary function called in the constructor.
        """
        
        self._train_mask_idxs = []
        self._test_mask_idxs = []
        self._val_mask_idxs = []

        list_eval_partition_filename = os.path.join(self._dataset_dir, "Eval", "list_eval_partition.txt")
        list_eval_partition_file = open(list_eval_partition_filename, 'r')

        for _ in range(2):
            line = list_eval_partition_file.readline()

        for img_idx, line in enumerate(list_eval_partition_file.readlines()):
            split_name = line.split()[3]
            if split_name == "train": self._train_mask_idxs.append(img_idx)
            elif split_name == "test": self._test_mask_idxs.append(img_idx)
            elif split_name == "val": self._val_mask_idxs.append(img_idx)

        list_eval_partition_file.close()


    def _write_list_eval_partition_aux_idxs_file(self):
        """
        Writes an auxiliary index file for faster data retrieval.
        Auxiliary function called in the constructor.
        Only called if the auxiliary index file does not exist.
        """

        # Create aux directory if non-existent

        aux_dirname = os.path.join(self._dataset_dir, "aux")
        if not os.path.exists(aux_dirname):
            os.mkdir(aux_dirname)

        # Compute image idxs

        image_bbox_aux_idxs_dict = {}

        image_bbox_filename = os.path.join(self._dataset_dir, "Anno", "list_bbox_consumer2shop.txt")
        image_bbox_file = open(image_bbox_filename, 'r')

        for _ in range(2):
            line = image_bbox_file.readline()

        for image_idx, line in enumerate(image_bbox_file.readlines()):
            image_filename = line.split()[0]
            image_bbox_aux_idxs_dict[image_filename] = image_idx

        image_bbox_file.close()

        # Compute item idxs

        item_attr_aux_idxs_dict = {}

        item_attr_filename = os.path.join(self._dataset_dir, "Anno", "list_attr_items.txt")
        item_attr_file = open(item_attr_filename, 'r')

        for _ in range(2):
            line = item_attr_file.readline()

        for item_idx, line in enumerate(item_attr_file.readlines()):
            item_id = line.split()[0]
            item_attr_aux_idxs_dict[item_id] = item_idx

        item_attr_file.close()

        # Write list_eval_partition_aux_idxs file

        list_eval_partition_filename = os.path.join(self._dataset_dir, "Eval", "list_eval_partition.txt")
        list_eval_partition_file = open(list_eval_partition_filename, 'r')

        list_eval_partition_aux_idxs_file = open(self._list_eval_partition_aux_idxs_filename, 'w')

        ## Copy first line from list_eval_partition file

        line = list_eval_partition_file.readline()
        list_eval_partition_aux_idxs_file.write(line)

        ## Skip second line from list_eval_partition file

        line = list_eval_partition_file.readline()

        ## Write list_eval_partition_aux_idxs file columns

        aux_line = "{:s} {:s} {:s}\n".format(
            "image_pair_bbox_idx_1",
            "image_pair_bbox_idx_2",
            "item_attr_idx"
        )

        list_eval_partition_aux_idxs_file.write(aux_line)

        ## Iterate over lines of list_eval_partition file

        for line in list_eval_partition_file.readlines():

            tkns = line.split()

            image_filename_1 = tkns[0]
            image_filename_2 = tkns[1]
            item_id = tkns[2]

            aux_line = "{:d} {:d} {:d}\n".format(
                image_bbox_aux_idxs_dict[image_filename_1],
                image_bbox_aux_idxs_dict[image_filename_2],
                item_attr_aux_idxs_dict[item_id]
            )

            list_eval_partition_aux_idxs_file.write(aux_line)

        list_eval_partition_file.close()
        list_eval_partition_aux_idxs_file.close()


    def _write_list_eval_partition_neg_aux_idxs_file(self):
        """
        TODO
        """

        # Create aux directory if non-existent

        aux_dirname = os.path.join(self._dataset_dir, "aux")
        if not os.path.exists(aux_dirname):
            os.mkdir(aux_dirname)

        # Compute image idxs

        image_bbox_aux_idxs_dict = {}

        image_bbox_filename = os.path.join(self._dataset_dir, "Anno", "list_bbox_consumer2shop.txt")
        image_bbox_file = open(image_bbox_filename, 'r')

        for _ in range(2):
            line = image_bbox_file.readline()

        for image_idx, line in enumerate(image_bbox_file.readlines()):
            image_filename = line.split()[0]
            image_bbox_aux_idxs_dict[image_filename] = image_idx

        image_bbox_file.close()

        image_bbox_aux_idxs_dict_keys_list = list(image_bbox_aux_idxs_dict.keys())

        # Compute item idxs

        item_attr_aux_idxs_dict = {}

        item_attr_filename = os.path.join(self._dataset_dir, "Anno", "list_attr_items.txt")
        item_attr_file = open(item_attr_filename, 'r')

        for _ in range(2):
            line = item_attr_file.readline()

        for item_idx, line in enumerate(item_attr_file.readlines()):
            item_id = line.split()[0]
            item_attr_aux_idxs_dict[item_id] = item_idx

        item_attr_file.close()

        # Write list_eval_partition_aux_idxs file

        list_eval_partition_filename = os.path.join(self._dataset_dir, "Eval", "list_eval_partition.txt")
        list_eval_partition_file = open(list_eval_partition_filename, 'r')

        list_eval_partition_neg_aux_idxs_file = open(self._list_eval_partition_neg_aux_idxs_filename, 'w')

        ## Copy first line from list_eval_partition file

        line = list_eval_partition_file.readline()
        list_eval_partition_neg_aux_idxs_file.write(line)

        ## Skip second line from list_eval_partition file

        line = list_eval_partition_file.readline()

        ## Write list_eval_partition_neg_aux_idxs file file columns

        aux_line = "{:s} {:s}\n".format(
            "neg_image_bbox_idx",
            "neg_item_idx"
        )

        list_eval_partition_neg_aux_idxs_file.write(aux_line)

        ## Iterate over lines of list_eval_partition file

        for line in list_eval_partition_file.readlines():

            tkns = line.split()

            item_id = tkns[2]

            neg_found = False
            while not neg_found:

                image_filename_neg = random.choice(image_bbox_aux_idxs_dict_keys_list)
                neg_item_id = image_filename_neg.split(os.sep)[3]

                neg_found = item_id != neg_item_id

            aux_line = "{:d} {:d}\n".format(
                image_bbox_aux_idxs_dict[image_filename_neg],
                item_attr_aux_idxs_dict[neg_item_id]
            )

            list_eval_partition_neg_aux_idxs_file.write(aux_line)

        list_eval_partition_file.close()
        list_eval_partition_neg_aux_idxs_file.close()


    def _compute_list_eval_partition_aux_idxs_cursors(self):
        """
        Computes file cursor values from the auxiliary index file.
        Auxiliary function called in the constructor.
        """

        self._list_eval_partition_aux_idxs_cursor_list = []

        list_eval_partition_aux_idxs_file = open(self._list_eval_partition_aux_idxs_filename, 'r')

        for _ in range(2):
            line = list_eval_partition_aux_idxs_file.readline()

        cursor = list_eval_partition_aux_idxs_file.tell()
        line = list_eval_partition_aux_idxs_file.readline()

        while line != "":
            
            self._list_eval_partition_aux_idxs_cursor_list.append(cursor)
            cursor = list_eval_partition_aux_idxs_file.tell()
            line = list_eval_partition_aux_idxs_file.readline()

        list_eval_partition_aux_idxs_file.close()


    def _compute_list_eval_partition_neg_aux_idxs_cursors(self):
        """
        TODO
        """

        self._list_eval_partition_neg_aux_idxs_cursor_list = []

        list_eval_partition_neg_aux_idxs_file = open(self._list_eval_partition_neg_aux_idxs_filename, 'r')

        for _ in range(2):
            line = list_eval_partition_neg_aux_idxs_file.readline()

        cursor = list_eval_partition_neg_aux_idxs_file.tell()
        line = list_eval_partition_neg_aux_idxs_file.readline()

        while line != "":
            
            self._list_eval_partition_neg_aux_idxs_cursor_list.append(cursor)
            cursor = list_eval_partition_neg_aux_idxs_file.tell()
            line = list_eval_partition_neg_aux_idxs_file.readline()

        list_eval_partition_neg_aux_idxs_file.close()



    def _compute_image_filename_bbox_cursors(self):
        """
        Computes file cursor values from the bbox file.
        Auxiliary function called in the constructor.
        """

        self._image_filename_bbox_cursor_list = []

        image_bbox_filename = os.path.join(self._dataset_dir, "Anno", "list_bbox_consumer2shop.txt")
        image_bbox_file = open(image_bbox_filename, 'r')

        for _ in range(2):
            line = image_bbox_file.readline()

        cursor = image_bbox_file.tell()
        line = image_bbox_file.readline()

        while line != "":

            self._image_filename_bbox_cursor_list.append(cursor)
            cursor = image_bbox_file.tell()
            line = image_bbox_file.readline()

        image_bbox_file.close()


    def _compute_item_attribute_cursors(self):
        """
        Computes file cursor values from the attribute file.
        Auxiliary function called in the constructor.
        """

        self._item_attribute_cursor_list = []

        item_attribute_filename = os.path.join(self._dataset_dir, "Anno", "list_attr_items.txt")
        item_attribute_file = open(item_attribute_filename, 'r')

        for _ in range(2):
            line = item_attribute_file.readline()

        cursor = item_attribute_file.tell()
        line = item_attribute_file.readline()

        while line != "":

            self._item_attribute_cursor_list.append(cursor)
            cursor = item_attribute_file.tell()
            line = item_attribute_file.readline()

        item_attribute_file.close()


    def _open_data_files(self):
        """
        Opens the required data files for reading.
        Auxiliary function called in the constructor.
        """

        self._list_eval_partition_aux_idxs_file = open(self._list_eval_partition_aux_idxs_filename, 'r')

        self._list_eval_partition_neg_aux_idxs_file = open(self._list_eval_partition_neg_aux_idxs_filename, 'r')

        image_bbox_filename = os.path.join(self._dataset_dir, "Anno", "list_bbox_consumer2shop.txt")
        self._image_bbox_file = open(image_bbox_filename, 'r')

        item_attribute_filename = os.path.join(self._dataset_dir, "Anno", "list_attr_items.txt")
        self._item_attribute_file = open(item_attribute_filename, 'r')


    def _initialize_locks(self):
        """
        Initializes multiprocessing locks for data files.
        Auxiliary function called in the constructor.
        """

        self._list_eval_partition_aux_idxs_file_lock = Lock()
        self._list_eval_partition_neg_aux_idxs_file_lock = Lock()
        self._image_bbox_file_lock = Lock()
        self._item_attribute_file_lock = Lock()


    def _read_list_eval_partition_aux_idxs_el(self, idx):
        """
        Reads auxiliary image and item idxs from the auxiliary train/test/val split file.
        Auxiliary function called when loading a data point.
        """
        
        with self._list_eval_partition_aux_idxs_file_lock:

            self._list_eval_partition_aux_idxs_file.seek(self._list_eval_partition_aux_idxs_cursor_list[idx])
            line = self._list_eval_partition_aux_idxs_file.readline()

        tkns = line.split()

        image_bbox_aux_idx_1 = int(tkns[0])
        image_bbox_aux_idx_2 = int(tkns[1])
        item_attr_idx = int(tkns[2])
        
        return image_bbox_aux_idx_1, image_bbox_aux_idx_2, item_attr_idx


    def _read_list_eval_partition_neg_aux_idxs_el(self, idx):
        """
        TODO
        """
        
        with self._list_eval_partition_neg_aux_idxs_file_lock:

            self._list_eval_partition_neg_aux_idxs_file.seek(self._list_eval_partition_neg_aux_idxs_cursor_list[idx])
            line = self._list_eval_partition_neg_aux_idxs_file.readline()

        tkns = line.split()

        image_bbox_aux_idx = int(tkns[0])
        item_attr_idx = int(tkns[1])
        
        return image_bbox_aux_idx, item_attr_idx
    

    def _read_image_filename_bbox_el_(self, image_bbox_idx):
        """
        Reads an image filename and bbox from the bbox file.
        Auxiliary function called when loading a data point.
        """
        
        with self._image_bbox_file_lock:

            self._image_bbox_file.seek(self._image_filename_bbox_cursor_list[image_bbox_idx])
            line = self._image_bbox_file.readline()

        tkns = line.split()

        image_filename = tkns[0]
        image_bbox = [int(tkn) for tkn in tkns[3:7]]
        
        return image_filename, image_bbox


    def _read_item_attribute_el(self, item_attr_idx):
        """
        Reads item attributes from the attribute file.
        Auxiliary function called when loading a data point.

        Can be used to separatedly read item attributes from the dataset to compute stats.
        """
        
        with self._item_attribute_file_lock:

            self._item_attribute_file.seek(self._item_attribute_cursor_list[item_attr_idx])
            line = self._item_attribute_file.readline()

        tkns = line.split()

        item_attributes = [int(tkn) for tkn in tkns[1:]]
        
        return item_attributes


    def _num_bytes(self):
        """
        Computes the memory usage to store all attributes of this object.
        Used only for debug purposes.

        :return num_bytes: int 
            Memory usage in bytes.
        """

        num_bytes = 0

        num_bytes += utils.mem.get_num_bytes(self._dataset_dir)
        num_bytes += utils.mem.get_num_bytes(self._img_transform)

        num_bytes += utils.mem.get_num_bytes(self._train_mask_idxs)
        num_bytes += utils.mem.get_num_bytes(self._test_mask_idxs)
        num_bytes += utils.mem.get_num_bytes(self._val_mask_idxs)

        num_bytes += utils.mem.get_num_bytes(self._attribute_name_list)
        num_bytes += utils.mem.get_num_bytes(self._supattribute_idx_list)
        num_bytes += utils.mem.get_num_bytes(self._supattribute_name_list)

        num_bytes += utils.mem.get_num_bytes(self._list_eval_partition_aux_idxs_cursor_list)
        num_bytes += utils.mem.get_num_bytes(self._list_eval_partition_neg_aux_idxs_cursor_list)
        num_bytes += utils.mem.get_num_bytes(self._image_filename_bbox_cursor_list)
        num_bytes += utils.mem.get_num_bytes(self._item_attribute_cursor_list)

        num_bytes += utils.mem.get_num_bytes(self._list_eval_partition_aux_idxs_file)
        num_bytes += utils.mem.get_num_bytes(self._list_eval_partition_neg_aux_idxs_file)
        num_bytes += utils.mem.get_num_bytes(self._image_bbox_file)
        num_bytes += utils.mem.get_num_bytes(self._item_attribute_file)

        num_bytes += utils.mem.get_num_bytes(self._list_eval_partition_aux_idxs_file_lock)
        num_bytes += utils.mem.get_num_bytes(self._list_eval_partition_neg_aux_idxs_file_lock)
        num_bytes += utils.mem.get_num_bytes(self._image_bbox_file_lock)
        num_bytes += utils.mem.get_num_bytes(self._item_attribute_file_lock)
        
        num_bytes += utils.mem.get_num_bytes(self._list_eval_partition_aux_idxs_filename)
        num_bytes += utils.mem.get_num_bytes(self._list_eval_partition_neg_aux_idxs_filename)

        return num_bytes


    def _is_pickable(self):
        """
        Attempts to pickle all attributes of this class and prints the respective status.
        Used only for debug purposes.
        """

        print("-- PICKLE STATUS START --")

        print("self._dataset_dir")
        print("  ", utils.pkl.pickle_test(self._dataset_dir))
        print("self._img_transform")
        print("  ", utils.pkl.pickle_test(self._img_transform))

        print("self._train_mask_idxs")
        print("  ", utils.pkl.pickle_test(self._train_mask_idxs))
        print("self._test_mask_idxs")
        print("  ", utils.pkl.pickle_test(self._test_mask_idxs))
        print("self._val_mask_idxs")
        print("  ", utils.pkl.pickle_test(self._val_mask_idxs))

        print("self._attribute_name_list")
        print("  ", utils.pkl.pickle_test(self._attribute_name_list))
        print("self._supattribute_idx_list")
        print("  ", utils.pkl.pickle_test(self._supattribute_idx_list))
        print("self._supattribute_name_list")
        print("  ", utils.pkl.pickle_test(self._supattribute_name_list))

        print("self._list_eval_partition_aux_idxs_cursor_list")
        print("  ", utils.pkl.pickle_test(self._list_eval_partition_aux_idxs_cursor_list))
        print("self._list_eval_partition_neg_aux_idxs_cursor_list")
        print("  ", utils.pkl.pickle_test(self._list_eval_partition_neg_aux_idxs_cursor_list))
        print("self._image_filename_bbox_cursor_list")
        print("  ", utils.pkl.pickle_test(self._image_filename_bbox_cursor_list))
        print("self._item_attribute_cursor_list")
        print("  ", utils.pkl.pickle_test(self._item_attribute_cursor_list))

        print("self._list_eval_partition_aux_idxs_file")
        print("  ", utils.pkl.pickle_test(self._list_eval_partition_aux_idxs_file))
        print("self._list_eval_partition_neg_aux_idxs_file")
        print("  ", utils.pkl.pickle_test(self._list_eval_partition_neg_aux_idxs_file))
        print("self._image_bbox_file")
        print("  ", utils.pkl.pickle_test(self._image_bbox_file))
        print("self._item_attribute_file")
        print("  ", utils.pkl.pickle_test(self._item_attribute_file))

        print("self._list_eval_partition_aux_idxs_file_lock")
        print("  ", utils.pkl.pickle_test(self._list_eval_partition_aux_idxs_file_lock))
        print("self._list_eval_partition_neg_aux_idxs_file_lock")
        print("  ", utils.pkl.pickle_test(self._list_eval_partition_neg_aux_idxs_file_lock))
        print("self._image_bbox_file_lock")
        print("  ", utils.pkl.pickle_test(self._image_bbox_file_lock))
        print("self._item_attribute_file_lock")
        print("  ", utils.pkl.pickle_test(self._item_attribute_file_lock))

        print("self._list_eval_partition_aux_idxs_filename")
        print("  ", utils.pkl.pickle_test(self._list_eval_partition_aux_idxs_filename))
        print("self._list_eval_partition_neg_aux_idxs_filename")
        print("  ", utils.pkl.pickle_test(self._list_eval_partition_neg_aux_idxs_filename))

        print("-- PICKLE STATUS END --")



class ConsToShopClothRetrBM_NEW(Dataset):
    """
    TODO
    """

    def __init__(
            self,
            dataset_dirname,
            img_transform = None,
            neg_img_filename_list_id = None
            ):
        """
        :param dataset_dirname: str
            Directory where the dataset is stored.
        :param img_transform: torch.Tensor -> torch.Tensor, default=None
            Transformation to apply to images.
        :param neg_aux_filename_id: str, default=None
            ID for the negative image idxs file. If no such file with this ID exists in the
            aux subdirectory of the dataset directory, a new one will be created. 
            If no ID is provided, a non-id file with random negative image idxs will be
            recalculated and used.
        """

        if not os.path.exists(dataset_dirname):
            raise ValueError("Directory {:s} does not exist".format(dataset_dirname))

        self._dataset_dirname = dataset_dirname
        self._img_transform = img_transform
        self._neg_img_filename_list_id = neg_img_filename_list_id

        self._compute_split_masks()

        self._compute_cloth_type_vars()
        self._compute_num_imgs()
        self._compute_num_img_pairs()

        self._write_img_filename_list_file()
        self._store_img_filename_codes()

        self._compute_img_filename_to_img_uid_dict()

        self._store_img_bbox_codes()

        self._store_img_pair_uid_codes()

        self._write_neg_img_filename_list_file()
        self._store_neg_img_uid_codes()

        self._delete_img_filename_to_img_uid_dict()


    def __len__(self):
        
        return self._num_img_pairs


    def __getitem__(self, idx):
        
        img_uid_1 = self._decode_img_uid(self._get_img_1_uid_code(idx))
        img_uid_2 = self._decode_img_uid(self._get_img_2_uid_code(idx))
        img_uid_3 = self._decode_img_uid(self._get_img_3_uid_code(idx))

        img_filename_1 = self._decode_img_filename(self._get_img_filename_code(img_uid_1))
        img_filename_2 = self._decode_img_filename(self._get_img_filename_code(img_uid_2))
        img_filename_3 = self._decode_img_filename(self._get_img_filename_code(img_uid_3))

        full_img_filename_1 = os.path.join(self._dataset_dirname, img_filename_1)
        img_1 = torchvision.io.read_image(full_img_filename_1)
        if img_1.size(dim=0) == 1: img_1 = img_1.repeat(3, 1, 1)

        full_img_filename_2 = os.path.join(self._dataset_dirname, img_filename_2)
        img_2 = torchvision.io.read_image(full_img_filename_2)
        if img_2.size(dim=0) == 1: img_2 = img_2.repeat(3, 1, 1)

        full_img_filename_3 = os.path.join(self._dataset_dirname, img_filename_3)
        img_3 = torchvision.io.read_image(full_img_filename_3)
        if img_3.size(dim=0) == 1: img_3 = img_3.repeat(3, 1, 1)

        img_bbox_1 = self._decode_img_bbox(self._get_img_bbox_code(img_uid_1))
        img_bbox_2 = self._decode_img_bbox(self._get_img_bbox_code(img_uid_2))
        img_bbox_3 = self._decode_img_bbox(self._get_img_bbox_code(img_uid_3))

        x1__1, y1__1, x2__1, y2__1 = img_bbox_1
        x1__2, y1__2, x2__2, y2__2 = img_bbox_2
        x1__3, y1__3, x2__3, y2__3 = img_bbox_3

        img_1 = img_1[:,y1__1:y2__1,x1__1:x2__1]
        img_2 = img_2[:,y1__2:y2__2,x1__2:x2__2]
        img_3 = img_3[:,y1__3:y2__3,x1__3:x2__3]

        if self._img_transform is not None:
            img_1 = self._img_transform(img_1)
            img_2 = self._img_transform(img_2)
            img_3 = self._img_transform(img_3)

        return img_1, img_2, img_3
    

    def get_split_mask_idxs(self, split_name):
        """
        Returns train/test/val split indices.

        :param split_name: str
            Split name. Can be "train", "test", or "val".
        
        :return: list
            A list with the split indices.
        """

        if split_name == "train": return self._train_mask_idxs
        elif split_name == "test": return self._test_mask_idxs
        elif split_name == "val": return self._val_mask_idxs


    ########
    # AUXILIARY INITIALIZATION METHODS
    ########


    def _compute_split_masks(self):
        """
        Reads train/test/val split indices.
        Auxiliary function called in the constructor.
        """
        
        self._train_mask_idxs = []
        self._test_mask_idxs = []
        self._val_mask_idxs = []

        list_eval_partition_filename = os.path.join(self._dataset_dirname, "Eval", "list_eval_partition.txt")
        list_eval_partition_file = open(list_eval_partition_filename, "r")

        for _ in range(2):
            line = list_eval_partition_file.readline()

        for img_idx, line in enumerate(list_eval_partition_file.readlines()):
            split_name = line.split()[3]
            if split_name == "train": self._train_mask_idxs.append(img_idx)
            elif split_name == "test": self._test_mask_idxs.append(img_idx)
            elif split_name == "val": self._val_mask_idxs.append(img_idx)

        self._train_mask_idxs = np.asarray(self._train_mask_idxs)
        self._test_mask_idxs = np.asarray(self._test_mask_idxs)
        self._val_mask_idxs = np.asarray(self._val_mask_idxs)

        list_eval_partition_file.close()


    def _compute_cloth_type_vars(self):

        img_dirname = os.path.join(self._dataset_dirname, "img")
        cloth_type_list = os.listdir(img_dirname)

        cloth_subtype_llist = []
        for cloth_type in cloth_type_list:
            cloth_dirname = os.path.join(img_dirname, cloth_type)
            cloth_subtype_llist.append(os.listdir(cloth_dirname))

        cloth_type_inv_dict = {cloth_type: idx for (idx, cloth_type) in enumerate(cloth_type_list)}
        cloth_subtype_inv_dict_list = [{cloth_subtype: idx for (idx, cloth_subtype) in enumerate(cloth_subtype_list)} for cloth_subtype_list in cloth_subtype_llist]

        self._cloth_type_list = cloth_type_list
        self._cloth_subtype_llist = cloth_subtype_llist
        self._cloth_type_inv_dict = cloth_type_inv_dict
        self._cloth_subtype_inv_dict_list = cloth_subtype_inv_dict_list

    
    def _compute_num_imgs(self):

        img_dirname = os.path.join(self._dataset_dirname, "img")

        self._num_imgs = 0
        for root, subdirs, files in os.walk(img_dirname):
            self._num_imgs += len(files)

    
    def _compute_num_img_pairs(self):

        list_eval_partition_filename = os.path.join(self._dataset_dirname, "Eval", "list_eval_partition.txt")
        list_eval_partition_file = open(list_eval_partition_filename, "r")

        self._num_img_pairs = int(list_eval_partition_file.readline()[:-1])

        list_eval_partition_file.close()


    def _write_img_filename_list_file(self):

        img_dirname = os.path.join(self._dataset_dirname, "img")
        img_filename_list_filename = os.path.join(self._dataset_dirname, "aux_NEW", "img_filename_list.txt")
        
        if os.path.exists(img_filename_list_filename): return
        
        img_filename_list_file = open(img_filename_list_filename, "w")

        img_filename_list_file.write("{:d}\n".format(self._num_imgs))
        img_filename_list_file.write("img_filename\n")

        for root, subdirs, files in os.walk(img_dirname):
            root_tkns = root.split(os.path.sep)
            partial_root = os.path.join(*root_tkns[-4:])
            for img_filename in files:
                img_filename = os.path.join(partial_root, img_filename)
                img_filename_list_file.write(img_filename + "\n")

        img_filename_list_file.close()


    def _store_img_filename_codes(self):

        self._img_filename_codes_arr = bytearray(4 * self._num_imgs)

        img_filename_list_filename = os.path.join(self._dataset_dirname, "aux_NEW", "img_filename_list.txt")
        img_filename_list_file = open(img_filename_list_filename, "r")

        for _ in range(2):
            img_filename_list_file.readline()

        for idx, line in enumerate(img_filename_list_file.readlines()):
            
            img_filename_code = self._encode_img_filename(line[:-1])          
            self._set_img_filename_code(idx, img_filename_code)

        img_filename_list_file.close()


    def _compute_img_filename_to_img_uid_dict(self):

        self._img_filename_to_img_uid_dict = {
            self._decode_img_filename(self._get_img_filename_code(idx)): idx
            for idx in range(self._num_imgs)
        }


    def _store_img_bbox_codes(self):

        self._img_bbox_codes_arr = bytearray(5 * self._num_imgs)

        img_bbox_filename = os.path.join(self._dataset_dirname, "Anno", "list_bbox_consumer2shop.txt")
        img_bbox_file = open(img_bbox_filename, "r")

        for _ in range(2):
            img_bbox_file.readline()

        for line in img_bbox_file.readlines():
            
            tkns = line.split()

            img_filename = tkns[0]
            img_filename_uid = self._img_filename_to_img_uid_dict[img_filename]

            img_bbox = [int(tkn) for tkn in tkns[-4:]]
            img_bbox_code = self._encode_img_bbox(img_bbox)
            self._set_img_bbox_code(img_filename_uid, img_bbox_code)

        img_bbox_file.close()


    def _store_img_pair_uid_codes(self):

        self._img_1_uid_codes_arr = bytearray(3 * self._num_imgs)
        self._img_2_uid_codes_arr = bytearray(3 * self._num_imgs)

        list_eval_partition_filename = os.path.join(self._dataset_dirname, "Eval", "list_eval_partition.txt")
        list_eval_partition_file = open(list_eval_partition_filename, "r")

        for _ in range(2):
            list_eval_partition_file.readline()

        for idx, line in enumerate(list_eval_partition_file.readlines()):
            
            tkns = line.split()

            img_1_uid = self._img_filename_to_img_uid_dict[tkns[0]]
            img_1_uid_code = self._encode_img_uid(img_1_uid)
            self._set_img_1_uid_code(idx, img_1_uid_code)

            img_2_uid = self._img_filename_to_img_uid_dict[tkns[1]]
            img_2_uid_code = self._encode_img_uid(img_2_uid)
            self._set_img_2_uid_code(idx, img_2_uid_code)

        list_eval_partition_file.close()


    def _write_neg_img_filename_list_file(self):

        neg_img_filename_list_filename_local = "neg_img_filename_list"
        if self._neg_img_filename_list_id is not None:
            neg_img_filename_list_filename_local += "__" + self._neg_img_filename_list_id
        neg_img_filename_list_filename_local += ".txt"
        neg_img_filename_list_filename = os.path.join(self._dataset_dirname, "aux_NEW", neg_img_filename_list_filename_local)
        
        if os.path.exists(neg_img_filename_list_filename) and self._neg_img_filename_list_id is not None: return

        neg_img_filename_list_file = open(neg_img_filename_list_filename, "w")

        neg_img_filename_list_file.write("{:d}\n".format(self._num_img_pairs))
        neg_img_filename_list_file.write("neg_img_filename\n")

        for img_pair_idx in range(self._num_img_pairs):

            img_1_uid = self._decode_img_uid(self._get_img_1_uid_code(img_pair_idx))
            img_1_filename = self._decode_img_filename(self._get_img_filename_code(img_1_uid))
            item_id = img_1_filename.split(os.path.sep)[3]

            neg_item_id = item_id
            while neg_item_id == item_id:
                neg_img_uid = random.randint(0, self._num_imgs - 1)
                neg_img_filename = self._decode_img_filename(self._get_img_filename_code(neg_img_uid))
                neg_item_id = neg_img_filename.split(os.path.sep)[3]
            
            neg_img_filename_list_file.write(neg_img_filename + "\n")

        neg_img_filename_list_file.close()


    def _store_neg_img_uid_codes(self):

        self._img_3_uid_codes_arr = bytearray(3 * self._num_imgs)

        neg_img_filename_list_filename_local = "neg_img_filename_list"
        if self._neg_img_filename_list_id is not None:
            neg_img_filename_list_filename_local += "__" + self._neg_img_filename_list_id
        neg_img_filename_list_filename_local += ".txt"
        neg_img_filename_list_filename = os.path.join(self._dataset_dirname, "aux_NEW", neg_img_filename_list_filename_local)
        
        neg_img_filename_list_file = open(neg_img_filename_list_filename, "r")

        for _ in range(2):
            neg_img_filename_list_file.readline()

        for idx, line in enumerate(neg_img_filename_list_file.readlines()):
            
            neg_img_uid = self._img_filename_to_img_uid_dict[line[:-1]]
            neg_img_uid_code = self._encode_img_uid(neg_img_uid)
            self._set_img_3_uid_code(idx, neg_img_uid_code)

        neg_img_filename_list_file.close()


    def _delete_img_filename_to_img_uid_dict(self):

        del self._img_filename_to_img_uid_dict


    ########
    # AUXILIARY CODE AND READ METHODS
    ########

    ####
    # IMAGE FILENAME
    ####


    def _encode_img_filename(self, img_filename):

        img_filename_tkns = img_filename.split(os.path.sep)
        img_filename_code = bytearray(4)

        # Cloth type and subtype

        cloth_type = img_filename_tkns[1]
        cloth_subtype = img_filename_tkns[2]

        cloth_type_zid = self._cloth_type_inv_dict[cloth_type]
        cloth_subtype_zid = self._cloth_subtype_inv_dict_list[cloth_type_zid][cloth_subtype]

        img_filename_code[0] = cloth_type_zid << 3 | cloth_subtype_zid

        # Item ID

        item_id = int(img_filename_tkns[3][3:])

        img_filename_code[1] = item_id >> 8
        img_filename_code[2] = item_id & 0b11111111

        # Domain type and image ID

        img_filename_tkns_4 = img_filename_tkns[4].split("_")
        domain_type = img_filename_tkns_4[0]
        img_id = int(img_filename_tkns_4[1][:2])

        img_filename_code[3] = img_id
        if domain_type == "shop": img_filename_code[3] += 0b01000000

        return img_filename_code
    

    def _decode_img_filename(self, img_filename_code):

        # Cloth type and subtype

        cloth_type_zid = (img_filename_code[0] & 0b00011000) >> 3
        cloth_subtype_zid = img_filename_code[0] & 0b00000111

        cloth_type = self._cloth_type_list[cloth_type_zid]
        cloth_subtype = self._cloth_subtype_llist[cloth_type_zid][cloth_subtype_zid]

        # Item ID

        item_id = img_filename_code[1] << 8 | img_filename_code[2]

        # Domain type and image ID

        img_id = img_filename_code[3] & 0b00111111
        domain_type = "shop" if img_filename_code[3] & 0b01000000 else "comsumer"

        # Build image filename

        img_filename = os.path.join(
            "img",
            cloth_type,
            cloth_subtype,
            "id_{:08d}".format(item_id),
            "{:s}_{:02d}.jpg".format(domain_type, img_id)
        )

        return img_filename
    

    def _set_img_filename_code(self, idx, img_filename_code):

        self._img_filename_codes_arr[4*idx:4*(idx+1)] = img_filename_code[:]
    

    def _get_img_filename_code(self, idx):

        return self._img_filename_codes_arr[4*idx:4*(idx+1)]
    

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
    # IMAGE UID
    ####


    def _encode_img_uid(self, img_uid):

        img_uid_code = bytearray(3)

        for idx in range(3):
            img_uid_code[2 - idx] = img_uid & 0b11111111
            img_uid >>= 8

        return img_uid_code


    def _decode_img_uid(self, img_uid_code):

        img_uid = 0

        for idx in range(3):
            img_uid <<= 8
            img_uid |= img_uid_code[idx]
            
        return img_uid


    def _set_img_1_uid_code(self, idx, img_uid_code):

        self._img_1_uid_codes_arr[3*idx:3*(idx+1)] = img_uid_code[:]
    

    def _get_img_1_uid_code(self, idx):

        return self._img_1_uid_codes_arr[3*idx:3*(idx+1)]


    def _set_img_2_uid_code(self, idx, img_uid_code):

        self._img_2_uid_codes_arr[3*idx:3*(idx+1)] = img_uid_code[:]
    

    def _get_img_2_uid_code(self, idx):

        return self._img_2_uid_codes_arr[3*idx:3*(idx+1)]


    def _set_img_3_uid_code(self, idx, img_uid_code):

        self._img_3_uid_codes_arr[3*idx:3*(idx+1)] = img_uid_code[:]
    

    def _get_img_3_uid_code(self, idx):

        return self._img_3_uid_codes_arr[3*idx:3*(idx+1)]


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

        num_bytes += utils.mem.get_num_bytes(self._dataset_dirname)
        num_bytes += utils.mem.get_num_bytes(self._img_transform)
        num_bytes += utils.mem.get_num_bytes(self._neg_img_filename_list_id)

        num_bytes += utils.mem.get_num_bytes(self._train_mask_idxs)
        num_bytes += utils.mem.get_num_bytes(self._test_mask_idxs)
        num_bytes += utils.mem.get_num_bytes(self._val_mask_idxs)

        num_bytes += utils.mem.get_num_bytes(self._cloth_type_list)
        num_bytes += utils.mem.get_num_bytes(self._cloth_subtype_llist)
        num_bytes += utils.mem.get_num_bytes(self._cloth_type_inv_dict)
        num_bytes += utils.mem.get_num_bytes(self._cloth_subtype_inv_dict_list)

        num_bytes += utils.mem.get_num_bytes(self._num_imgs)
        num_bytes += utils.mem.get_num_bytes(self._num_img_pairs)

        num_bytes += utils.mem.get_num_bytes(self._img_filename_codes_arr)
        num_bytes += utils.mem.get_num_bytes(self._img_bbox_codes_arr)

        num_bytes += utils.mem.get_num_bytes(self._img_1_uid_codes_arr)
        num_bytes += utils.mem.get_num_bytes(self._img_2_uid_codes_arr)
        num_bytes += utils.mem.get_num_bytes(self._img_3_uid_codes_arr)

        return num_bytes


    def _is_pickable(self):
        """
        Attempts to pickle all attributes of this class and prints the respective status.
        Used only for debug purposes.
        """

        print("-- PICKLE STATUS START --")

        print("self._dataset_dirname")
        print("  ", utils.pkl.pickle_test(self._dataset_dirname))
        print("self._img_transform")
        print("  ", utils.pkl.pickle_test(self._img_transform))
        print("self._neg_img_filename_list_id")
        print("  ", utils.pkl.pickle_test(self._neg_img_filename_list_id))

        print("self._train_mask_idxs")
        print("  ", utils.pkl.pickle_test(self._train_mask_idxs))
        print("self._test_mask_idxs")
        print("  ", utils.pkl.pickle_test(self._test_mask_idxs))
        print("self._val_mask_idxs")
        print("  ", utils.pkl.pickle_test(self._val_mask_idxs))

        print("self._cloth_type_list")
        print("  ", utils.pkl.pickle_test(self._cloth_type_list))
        print("self._cloth_subtype_llist")
        print("  ", utils.pkl.pickle_test(self._cloth_subtype_llist))
        print("self._cloth_type_inv_dict")
        print("  ", utils.pkl.pickle_test(self._cloth_type_inv_dict))
        print("self._cloth_subtype_inv_dict_list")
        print("  ", utils.pkl.pickle_test(self._cloth_subtype_inv_dict_list))
        
        print("self._num_imgs")
        print("  ", utils.pkl.pickle_test(self._num_imgs))
        print("self._num_img_pairs")
        print("  ", utils.pkl.pickle_test(self._num_img_pairs))
        
        print("self._img_filename_codes_arr")
        print("  ", utils.pkl.pickle_test(self._img_filename_codes_arr))
        print("self._img_bbox_codes_arr")
        print("  ", utils.pkl.pickle_test(self._img_bbox_codes_arr))

        print("self._img_1_uid_codes_arr")
        print("  ", utils.pkl.pickle_test(self._img_1_uid_codes_arr))
        print("self._img_2_uid_codes_arr")
        print("  ", utils.pkl.pickle_test(self._img_2_uid_codes_arr))
        print("self._img_3_uid_codes_arr")
        print("  ", utils.pkl.pickle_test(self._img_3_uid_codes_arr))

        print("-- PICKLE STATUS END --")
