import torch
import torchvision

import torchvision.io
from torch.utils.data import Dataset

import os
from multiprocessing import Lock

import utils

import random


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
            image_transform = None
            ):
        """
        :param dataset_dir: str
            Directory where the dataset is stored.
        :param image_transform: torch.Tensor -> torch.Tensor
            Transformation to apply to images.
        """

        if not os.path.exists(dataset_dir):
            raise ValueError("Directory {:s} does not exist".format(dataset_dir))

        self._dataset_dir = dataset_dir
        self._image_transform = image_transform

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

        if self._image_transform is not None:
            image = self._image_transform(image)

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

        num_bytes += utils.get_num_bytes(self._dataset_dir)
        num_bytes += utils.get_num_bytes(self._image_transform)

        num_bytes += utils.get_num_bytes(self._train_mask_idxs)
        num_bytes += utils.get_num_bytes(self._test_mask_idxs)
        num_bytes += utils.get_num_bytes(self._val_mask_idxs)

        num_bytes += utils.get_num_bytes(self._category_name_list)
        num_bytes += utils.get_num_bytes(self._supcategory_idx_list)
        num_bytes += utils.get_num_bytes(self._supcategory_name_list)

        num_bytes += utils.get_num_bytes(self._attribute_name_list)
        num_bytes += utils.get_num_bytes(self._supattribute_idx_list)
        num_bytes += utils.get_num_bytes(self._supattribute_name_list)

        num_bytes += utils.get_num_bytes(self._list_eval_partition_cursor_list)
        num_bytes += utils.get_num_bytes(self._image_bbox_cursor_list)
        num_bytes += utils.get_num_bytes(self._image_category_cursor_list)
        num_bytes += utils.get_num_bytes(self._image_attribute_cursor_list)

        num_bytes += utils.get_num_bytes(self._list_eval_partition_file)
        num_bytes += utils.get_num_bytes(self._image_bbox_file)
        num_bytes += utils.get_num_bytes(self._image_category_file)
        num_bytes += utils.get_num_bytes(self._image_attribute_file)

        num_bytes += utils.get_num_bytes(self._list_eval_partition_file_lock)
        num_bytes += utils.get_num_bytes(self._image_bbox_file_lock)
        num_bytes += utils.get_num_bytes(self._image_category_file_lock)
        num_bytes += utils.get_num_bytes(self._image_attribute_file_lock)

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
            image_transform = None
            ):
        """
        :param dataset_dir: str
            Directory where the dataset is stored.
        :param image_transform: torch.Tensor -> torch.Tensor
            Transformation to apply to images.
        """

        if not os.path.exists(dataset_dir):
            raise ValueError("Directory {:s} does not exist".format(dataset_dir))

        self._dataset_dir = dataset_dir
        self._image_transform = image_transform

        self._load_attribute_metadata()
        self._load_split_masks()
        
        list_eval_partition_aux_idxs_filename = os.path.join(self._dataset_dir, "aux", "list_eval_partition_aux_idxs.txt")
        if not os.path.exists(list_eval_partition_aux_idxs_filename):
            self._write_list_eval_partition_aux_idxs_file()

        self._compute_list_eval_partition_cursors()
        self._compute_list_eval_partition_aux_idxs_cursors()
        self._compute_image_bbox_cursors()
        self._compute_item_attribute_cursors()

        self._open_data_files()
        self._initialize_locks()


    def __len__(self):

        return len(self._list_eval_partition_cursor_list)


    def __getitem__(self, idx):

        image_filename_1, image_filename_2, item_id = self._read_list_eval_partition_el(idx)

        full_image_filename_1 = os.path.join(self._dataset_dir, image_filename_1)
        image_1 = torchvision.io.read_image(full_image_filename_1)
        if image_1.size(dim=0) == 1: image_1 = image_1.repeat(3, 1, 1)

        full_image_filename_2 = os.path.join(self._dataset_dir, image_filename_2)
        image_2 = torchvision.io.read_image(full_image_filename_2)
        if image_2.size(dim=0) == 1: image_2 = image_2.repeat(3, 1, 1)

        image_bbox_idx_1, image_bbox_idx_2, item_attr_idx = self._read_list_eval_partition_idxs_el(idx)

        x1__1, y1__1, x2__1, y2__1 = self._read_image_bbox_el(image_bbox_idx_1)
        x1__2, y1__2, x2__2, y2__2 = self._read_image_bbox_el(image_bbox_idx_2)

        neg_found = False
        while not neg_found:

            neg_image_bbox_idx = random.randint(0, len(self._image_bbox_cursor_list) - 1)
            image_filename_3, bbox_3 = self._read_image_filename_bbox_el_(neg_image_bbox_idx)
            
            if image_filename_3.split(os.sep)[3] != item_id:
                neg_found = True

        full_image_filename_3 = os.path.join(self._dataset_dir, image_filename_3)
        image_3 = torchvision.io.read_image(full_image_filename_3)
        if image_3.size(dim=0) == 1: image_3 = image_3.repeat(3, 1, 1)

        x1__3, y1__3, x2__3, y2__3 = bbox_3[0], bbox_3[1], bbox_3[2], bbox_3[3]

        image_1 = image_1[:,y1__1:y2__1,x1__1:x2__1]
        image_2 = image_2[:,y1__2:y2__2,x1__2:x2__2]
        image_3 = image_3[:,y1__3:y2__3,x1__3:x2__3]

        if self._image_transform is not None:
            image_1 = self._image_transform(image_1)
            image_2 = self._image_transform(image_2)
            image_3 = self._image_transform(image_3)

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

        list_eval_partition_aux_idxs_filename = os.path.join(self._dataset_dir, "aux", "list_eval_partition_aux_idxs.txt")
        list_eval_partition_aux_idxs_file = open(list_eval_partition_aux_idxs_filename, 'w')

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


    def _compute_list_eval_partition_aux_idxs_cursors(self):
        """
        Computes file cursor values from the auxiliary index file.
        Auxiliary function called in the constructor.
        """

        self._list_eval_partition_idxs_cursor_list = []

        list_eval_partition_aux_idxs_filename = os.path.join(self._dataset_dir, "aux", "list_eval_partition_aux_idxs.txt")
        list_eval_partition_aux_idxs_file = open(list_eval_partition_aux_idxs_filename, 'r')

        for _ in range(2):
            line = list_eval_partition_aux_idxs_file.readline()

        cursor = list_eval_partition_aux_idxs_file.tell()
        line = list_eval_partition_aux_idxs_file.readline()

        while line != "":
            
            self._list_eval_partition_idxs_cursor_list.append(cursor)
            cursor = list_eval_partition_aux_idxs_file.tell()
            line = list_eval_partition_aux_idxs_file.readline()

        list_eval_partition_aux_idxs_file.close()


    def _compute_image_bbox_cursors(self):
        """
        Computes file cursor values from the bbox file.
        Auxiliary function called in the constructor.
        """

        self._image_bbox_cursor_list = []

        image_bbox_filename = os.path.join(self._dataset_dir, "Anno", "list_bbox_consumer2shop.txt")
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

        list_eval_partition_filename = os.path.join(self._dataset_dir, "Eval", "list_eval_partition.txt")
        self._list_eval_partition_file = open(list_eval_partition_filename, 'r')

        list_eval_partition_aux_idxs_filename = os.path.join(self._dataset_dir, "aux", "list_eval_partition_aux_idxs.txt")
        self._list_eval_partition_aux_idxs_file = open(list_eval_partition_aux_idxs_filename, 'r')

        image_bbox_filename = os.path.join(self._dataset_dir, "Anno", "list_bbox_consumer2shop.txt")
        self._image_bbox_file = open(image_bbox_filename, 'r')

        item_attribute_filename = os.path.join(self._dataset_dir, "Anno", "list_attr_items.txt")
        self._item_attribute_file = open(item_attribute_filename, 'r')


    def _initialize_locks(self):
        """
        Initializes multiprocessing locks for data files.
        Auxiliary function called in the constructor.
        """

        self._list_eval_partition_file_lock = Lock()
        self._list_eval_partition_aux_idxs_file_lock = Lock()
        self._image_bbox_file_lock = Lock()
        self._item_attribute_file_lock = Lock()


    def _read_list_eval_partition_el(self, idx):
        """
        Reads image filenames and an item id from the train/test/val split file.
        Auxiliary function called when loading a data point.
        """
        
        with self._list_eval_partition_file_lock:

            self._list_eval_partition_file.seek(self._list_eval_partition_cursor_list[idx])
            line = self._list_eval_partition_file.readline()

        tkns = line.split()

        image_filename_1 = tkns[0]
        image_filename_2 = tkns[1]
        item_id = tkns[2]
        
        return image_filename_1, image_filename_2, item_id


    def _read_list_eval_partition_idxs_el(self, idx):
        """
        Reads auxiliary image and item idxs from the auxiliary train/test/val split file.
        Auxiliary function called when loading a data point.
        """
        
        with self._list_eval_partition_aux_idxs_file_lock:

            self._list_eval_partition_aux_idxs_file.seek(self._list_eval_partition_idxs_cursor_list[idx])
            line = self._list_eval_partition_aux_idxs_file.readline()

        tkns = line.split()

        image_bbox_aux_idxs_1 = int(tkns[0])
        image_bbox_aux_idxs_2 = int(tkns[1])
        item_attr_idx = int(tkns[2])
        
        return image_bbox_aux_idxs_1, image_bbox_aux_idxs_2, item_attr_idx


    def _read_image_bbox_el(self, image_bbox_idx):
        """
        Reads an image bbox from the bbox file.
        Auxiliary function called when loading a data point.
        """
        
        with self._image_bbox_file_lock:

            self._image_bbox_file.seek(self._image_bbox_cursor_list[image_bbox_idx])
            line = self._image_bbox_file.readline()

        tkns = line.split()

        image_bbox = [int(tkn) for tkn in tkns[3:7]]
        
        return image_bbox
    

    def _read_image_filename_bbox_el_(self, image_bbox_idx):
        """
        Reads an image filename and bbox from the bbox file.
        Auxiliary function called when loading a data point.
        """
        
        with self._image_bbox_file_lock:

            self._image_bbox_file.seek(self._image_bbox_cursor_list[image_bbox_idx])
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

        num_bytes += utils.get_num_bytes(self._dataset_dir)
        num_bytes += utils.get_num_bytes(self._image_transform)

        num_bytes += utils.get_num_bytes(self._train_mask_idxs)
        num_bytes += utils.get_num_bytes(self._test_mask_idxs)
        num_bytes += utils.get_num_bytes(self._val_mask_idxs)

        num_bytes += utils.get_num_bytes(self._attribute_name_list)
        num_bytes += utils.get_num_bytes(self._supattribute_idx_list)
        num_bytes += utils.get_num_bytes(self._supattribute_name_list)

        num_bytes += utils.get_num_bytes(self._list_eval_partition_cursor_list)
        num_bytes += utils.get_num_bytes(self._list_eval_partition_idxs_cursor_list)
        num_bytes += utils.get_num_bytes(self._image_bbox_cursor_list)
        num_bytes += utils.get_num_bytes(self._item_attribute_cursor_list)

        num_bytes += utils.get_num_bytes(self._list_eval_partition_file)
        num_bytes += utils.get_num_bytes(self._list_eval_partition_aux_idxs_file)
        num_bytes += utils.get_num_bytes(self._image_bbox_file)
        num_bytes += utils.get_num_bytes(self._item_attribute_file)

        num_bytes += utils.get_num_bytes(self._list_eval_partition_file_lock)
        num_bytes += utils.get_num_bytes(self._list_eval_partition_aux_idxs_file_lock)
        num_bytes += utils.get_num_bytes(self._image_bbox_file_lock)
        num_bytes += utils.get_num_bytes(self._item_attribute_file_lock)

        return num_bytes