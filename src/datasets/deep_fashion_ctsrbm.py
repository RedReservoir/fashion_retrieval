import torch
import torchvision

from torch.utils.data import Dataset
import torchvision.io

import os
import random

import numpy as np

import src.utils.memory
import src.utils.pkl



class ConsToShopClothRetrBmkDataset(Dataset):
    """
    Dataset class for the DeepFashion Consumer-to-shop Clothes Retrieval Benchmark dataset.
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

        self._compute_cloth_type_vars()

        self._compute_num_imgs()
        self._compute_num_img_pairs()

        self._compute_masks()

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
        """
        Returns a dataset item.

        :param idx: int
            Index of the dataset item.

        :return img_anc: torch.tensor
            Anchor image from the shop domain.
        :return img_pos: torch.tensor
            Positive image from the consumer domain (same clothes as img_anc).
        :return img_neg: torch.tensor
            Negative image from the consumer domain (different clothes as img_anc).
        """
        
        img_anc_uid = self._decode_img_uid(self._get_img_anc_uid_code(idx))
        img_pos_uid = self._decode_img_uid(self._get_img_pos_uid_code(idx))
        img_neg_uid = self._decode_img_uid(self._get_img_neg_uid_code(idx))

        img_anc_filename = self._decode_img_filename(self._get_img_filename_code(img_anc_uid))
        img_pos_filename = self._decode_img_filename(self._get_img_filename_code(img_pos_uid))
        img_neg_filename = self._decode_img_filename(self._get_img_filename_code(img_neg_uid))

        full_img_anc_filename = os.path.join(self._dataset_dirname, img_anc_filename)
        img_anc = torchvision.io.read_image(full_img_anc_filename)
        if img_anc.size(dim=0) == 1: img_anc = img_anc.repeat(3, 1, 1)

        full_img_pos_filename = os.path.join(self._dataset_dirname, img_pos_filename)
        img_pos = torchvision.io.read_image(full_img_pos_filename)
        if img_pos.size(dim=0) == 1: img_pos = img_pos.repeat(3, 1, 1)

        full_img_neg_filename = os.path.join(self._dataset_dirname, img_neg_filename)
        img_neg = torchvision.io.read_image(full_img_neg_filename)
        if img_neg.size(dim=0) == 1: img_neg = img_neg.repeat(3, 1, 1)

        img_anc_bbox = self._decode_img_bbox(self._get_img_bbox_code(img_anc_uid))
        img_pos_bbox = self._decode_img_bbox(self._get_img_bbox_code(img_pos_uid))
        img_neg_bbox = self._decode_img_bbox(self._get_img_bbox_code(img_neg_uid))

        x1__1, y1__1, x2__1, y2__1 = img_anc_bbox
        x1__2, y1__2, x2__2, y2__2 = img_pos_bbox
        x1__3, y1__3, x2__3, y2__3 = img_neg_bbox

        img_anc = img_anc[:,y1__1:y2__1,x1__1:x2__1]
        img_pos = img_pos[:,y1__2:y2__2,x1__2:x2__2]
        img_neg = img_neg[:,y1__3:y2__3,x1__3:x2__3]

        if self._img_transform is not None:
            img_anc = self._img_transform(img_anc)
            img_pos = self._img_transform(img_pos)
            img_neg = self._img_transform(img_neg)

        return img_anc, img_pos, img_neg
    

    def get_subset_indices(self, split=None):
        """
        Returns dataset subset indices.

        :param split: str, optional
            Split name. Can be "train", "test", or "val".
            Split will not be taken into account if not provided.
        
        :return: np.ndarray
            An array with the split indices.
        """

        mask = np.full(shape=(self._num_img_pairs), fill_value=True)

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


    def _compute_cloth_type_vars(self):
        """
        Auxiliary function called in the constructor.

        Computes cloth types and subtypes from the `img` directory.
        Cloth type attributes:
          - `_cloth_type_list`: List of the different cloth types.
          - `_cloth_type_inv_dict`: Dict that returns an idx given a cloth type.
        Cloth subtype attributes:
          - `_cloth_subtype_llist`: 2D list of the different cloth subtypes.
          - `_cloth_subtype_inv_dict_list`: List of dicts that return an idx given a cloth subtype.
        """

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
        """
        Auxiliary function called in the constructor.
        
        Computes the number of images in the dataset.
        The number of images is stored in attribute `_num_imgs`.
        If the `aux/img_filename_list.txt` exists, the number is read from the file.
        Otherwise, the number is computed by counting the number of images in the `img` directory.
        """

        img_dirname = os.path.join(self._dataset_dirname, "img")
        img_filename_list_filename = os.path.join(self._dataset_dirname, "aux", "img_filename_list.txt")
        
        if os.path.exists(img_filename_list_filename):

            img_filename_list_file = open(img_filename_list_filename, "r")

            self._num_imgs = int(img_filename_list_file.readline()[:-1])

            img_filename_list_file.close()

        else:

            self._num_imgs = 0
            for root, subdirs, files in os.walk(img_dirname):
                self._num_imgs += len(files)

    
    def _compute_num_img_pairs(self):
        """
        Auxiliary function called in the constructor.
        
        Computes the number of image pairs in the dataset.
        The number of image pairs is stored in attribute `_num_img_pairs`.
        The number is read from file `Eval/list_eval_partition.txt` file.
        """

        list_eval_partition_filename = os.path.join(self._dataset_dirname, "Eval", "list_eval_partition.txt")
        list_eval_partition_file = open(list_eval_partition_filename, "r")

        self._num_img_pairs = int(list_eval_partition_file.readline()[:-1])

        list_eval_partition_file.close()


    def _compute_masks(self):
        """
        Auxiliary function called in the constructor.

        Computes train/test/val split masks.
        Indices are stored in the following attributes:
          - `_split_num_mask`, where train/val/test is 0/1/2
        """

        # Split mask

        self._split_num_mask = np.empty(shape=(self._num_img_pairs), dtype=int)

        list_eval_partition_filename = os.path.join(self._dataset_dirname, "Eval", "list_eval_partition.txt")
        list_eval_partition_file = open(list_eval_partition_filename, "r")

        for _ in range(2):
            line = list_eval_partition_file.readline()

        for img_idx, line in enumerate(list_eval_partition_file.readlines()):

            split_name = line.split()[3]
            
            if split_name == "train": self._split_num_mask[img_idx] = 0
            elif split_name == "val": self._split_num_mask[img_idx] = 1
            elif split_name == "test": self._split_num_mask[img_idx] = 2

        list_eval_partition_file.close()


    def _write_img_filename_list_file(self):
        """
        Auxiliary function called in the constructor.
        
        Writes a complete list of all images from the dataset in the `aux/img_filename_list.txt` file.
        The file is not written if it already exists.
        The image list is obtained by traversing the `img` directory with `os.walk`.
        """

        img_dirname = os.path.join(self._dataset_dirname, "img")
        img_filename_list_filename = os.path.join(self._dataset_dirname, "aux", "img_filename_list.txt")
        
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
        """
        Auxiliary function called in the constructor.
        
        Encodes image filenames and stores them in the `_img_filename_codes_arr` attribute.
        Image filenames are read from the `aux/img_filename_list.txt` file.
        """

        self._img_filename_codes_arr = bytearray(4 * self._num_imgs)

        img_filename_list_filename = os.path.join(self._dataset_dirname, "aux", "img_filename_list.txt")
        img_filename_list_file = open(img_filename_list_filename, "r")

        for _ in range(2):
            img_filename_list_file.readline()

        for idx, line in enumerate(img_filename_list_file.readlines()):
            
            img_filename_code = self._encode_img_filename(line[:-1])          
            self._set_img_filename_code(idx, img_filename_code)

        img_filename_list_file.close()


    def _compute_img_filename_to_img_uid_dict(self):
        """
        Auxiliary function called in the constructor.
        
        Computes a dict that maps image filenames to an image uid and stores it in the
        `_img_filename_to_img_uid_dict` attribute.
        The dict serves as a temporary auxiliary variable.
        The dict is deleted afterwards in `_delete_img_filename_to_img_uid_dict`.
        """

        self._img_filename_to_img_uid_dict = {
            self._decode_img_filename(self._get_img_filename_code(idx)): idx
            for idx in range(self._num_imgs)
        }


    def _store_img_bbox_codes(self):
        """
        Auxiliary function called in the constructor.
        
        Encodes image bboxes and stores them in the `_img_bbox_codes_arr` attribute.
        Image bboxes are read from the `Anno/list_bbox_consumer2shop.txt` file.
        """

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
        """
        Auxiliary function called in the constructor.
        
        Encodes anchor and positive image uids from image pairs and stores them in the
        `_img_anc_uid_codes_arr` and `_img_pos_uid_codes_arr` attributes.
        Image pairs are read from the `Eval/list_eval_partition.txt` file.
        """

        self._img_anc_uid_codes_arr = bytearray(3 * self._num_imgs)
        self._img_pos_uid_codes_arr = bytearray(3 * self._num_imgs)

        list_eval_partition_filename = os.path.join(self._dataset_dirname, "Eval", "list_eval_partition.txt")
        list_eval_partition_file = open(list_eval_partition_filename, "r")

        for _ in range(2):
            list_eval_partition_file.readline()

        for idx, line in enumerate(list_eval_partition_file.readlines()):
            
            tkns = line.split()

            img_anc_uid = self._img_filename_to_img_uid_dict[tkns[0]]
            img_anc_uid_code = self._encode_img_uid(img_anc_uid)
            self._set_img_anc_uid_code(idx, img_anc_uid_code)

            img_pos_uid = self._img_filename_to_img_uid_dict[tkns[1]]
            img_pos_uid_code = self._encode_img_uid(img_pos_uid)
            self._set_img_pos_uid_code(idx, img_pos_uid_code)

        list_eval_partition_file.close()


    def _write_neg_img_filename_list_file(self):
        """
        Auxiliary function called in the constructor.

        Writes a list of negative images for image pairs to the
        `aux/neg_img_filename_list__xxx.txt` file, where `xxx` is the value in the
        `_neg_img_filename_list_id` attribute from the constructor.
        
        For each image pair, a random negative image is selected such that:
          - The negative image item id is different than the anchor image item id.
          - The negative image is from the sample split than the anchor image item id.
          - The negative image is from a different domain than the anchor image (all anchor
          images are from the consumer domain, so negative images are from the shop domain).
        """

        # Shop/consumer and item ID masks

        domain_num_mask = np.empty(shape=(self._num_imgs), dtype=int)
        item_id_mask = np.empty(shape=(self._num_imgs), dtype=int)

        img_filename_list_filename = os.path.join(self._dataset_dirname, "aux", "img_filename_list.txt")
        img_filename_list_file = open(img_filename_list_filename, "r")

        for _ in range(2):
            img_filename_list_file.readline()

        for idx, line in enumerate(img_filename_list_file.readlines()):
            
            tkns = line.split(os.path.sep)
            domain_num = 0 if tkns[4][:-8] == "shop" else 1
            item_id = int(tkns[3][3:])

            domain_num_mask[idx] = domain_num
            item_id_mask[idx] = item_id

        img_filename_list_file.close()

        # Split mask

        split_num_mask = np.empty(shape=(self._num_imgs), dtype=int)

        item_id_set = set()

        list_eval_partition_filename = os.path.join(self._dataset_dirname, "Eval", "list_eval_partition.txt")
        list_eval_partition_file = open(list_eval_partition_filename, "r")

        for _ in range(2):
            line = list_eval_partition_file.readline()

        for line in list_eval_partition_file.readlines():

            tkns = line.split()
            item_id = int(tkns[2][3:])
            split_name = tkns[3]
            
            if item_id not in item_id_set:

                if split_name == "train": split_num_mask[item_id_mask == item_id] = 0
                elif split_name == "val": split_num_mask[item_id_mask == item_id] = 1
                elif split_name == "test": split_num_mask[item_id_mask == item_id] = 2

                item_id_set.add(item_id)

        # Writing

        rd_mask = np.empty(shape=(self._num_imgs), dtype=int)

        neg_img_filename_list_filename_local = "neg_img_filename_list"
        if self._neg_img_filename_list_id is not None:
            neg_img_filename_list_filename_local += "__" + self._neg_img_filename_list_id
        neg_img_filename_list_filename_local += ".txt"
        neg_img_filename_list_filename = os.path.join(self._dataset_dirname, "aux", neg_img_filename_list_filename_local)

        if os.path.exists(neg_img_filename_list_filename) and self._neg_img_filename_list_id is not None: return

        neg_img_filename_list_file = open(neg_img_filename_list_filename, "w")

        neg_img_filename_list_file.write("{:d}\n".format(self._num_img_pairs))
        neg_img_filename_list_file.write("neg_img_filename\n")

        for img_pair_idx in range(self._num_img_pairs):

            img_anc_uid = self._decode_img_uid(self._get_img_anc_uid_code(img_pair_idx))

            split_num = split_num_mask[img_anc_uid]
            item_id = item_id_mask[img_anc_uid]

            rd_mask[:] = True
            rd_mask[domain_num_mask == 1] = False
            rd_mask[split_num_mask != split_num] = False
            rd_mask[item_id_mask == item_id] = False

            neg_img_uid = np.random.choice(np.argwhere(rd_mask).flatten())
            neg_img_filename = self._decode_img_filename(self._get_img_filename_code(neg_img_uid))
            
            neg_img_filename_list_file.write(neg_img_filename + "\n")

        neg_img_filename_list_file.close()


    def _store_neg_img_uid_codes(self):
        """
        Auxiliary function called in the constructor.
        
        Encodes negative image uids from image pairs and stores them in the
        `_img_neg_uid_codes_arr` attributes.
        Negative image filenames are read from the `aux/neg_img_filename_list__xxx.txt` file,
        where `xxx` is the value in the `_neg_img_filename_list_id` attribute from the constructor.
        """

        self._img_neg_uid_codes_arr = bytearray(3 * self._num_imgs)

        neg_img_filename_list_filename_local = "neg_img_filename_list"
        if self._neg_img_filename_list_id is not None:
            neg_img_filename_list_filename_local += "__" + self._neg_img_filename_list_id
        neg_img_filename_list_filename_local += ".txt"
        neg_img_filename_list_filename = os.path.join(self._dataset_dirname, "aux", neg_img_filename_list_filename_local)
        
        neg_img_filename_list_file = open(neg_img_filename_list_filename, "r")

        for _ in range(2):
            neg_img_filename_list_file.readline()

        for idx, line in enumerate(neg_img_filename_list_file.readlines()):
            
            neg_img_uid = self._img_filename_to_img_uid_dict[line[:-1]]
            neg_img_uid_code = self._encode_img_uid(neg_img_uid)
            self._set_img_neg_uid_code(idx, neg_img_uid_code)

        neg_img_filename_list_file.close()


    def _delete_img_filename_to_img_uid_dict(self):
        """
        Auxiliary function called in the constructor.
        
        Deletes the `_img_filename_to_img_uid_dict` attribute created in
        `_compute_img_filename_to_img_uid_dict`.
        """

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


    def _set_img_anc_uid_code(self, idx, img_uid_code):

        self._img_anc_uid_codes_arr[3*idx:3*(idx+1)] = img_uid_code[:]
    

    def _get_img_anc_uid_code(self, idx):

        return self._img_anc_uid_codes_arr[3*idx:3*(idx+1)]


    def _set_img_pos_uid_code(self, idx, img_uid_code):

        self._img_pos_uid_codes_arr[3*idx:3*(idx+1)] = img_uid_code[:]
    

    def _get_img_pos_uid_code(self, idx):

        return self._img_pos_uid_codes_arr[3*idx:3*(idx+1)]


    def _set_img_neg_uid_code(self, idx, img_uid_code):

        self._img_neg_uid_codes_arr[3*idx:3*(idx+1)] = img_uid_code[:]
    

    def _get_img_neg_uid_code(self, idx):

        return self._img_neg_uid_codes_arr[3*idx:3*(idx+1)]


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
        num_bytes += src.utils.memory.get_num_bytes(self._neg_img_filename_list_id)

        num_bytes += src.utils.memory.get_num_bytes(self._split_num_mask)

        num_bytes += src.utils.memory.get_num_bytes(self._cloth_type_list)
        num_bytes += src.utils.memory.get_num_bytes(self._cloth_subtype_llist)
        num_bytes += src.utils.memory.get_num_bytes(self._cloth_type_inv_dict)
        num_bytes += src.utils.memory.get_num_bytes(self._cloth_subtype_inv_dict_list)

        num_bytes += src.utils.memory.get_num_bytes(self._num_imgs)
        num_bytes += src.utils.memory.get_num_bytes(self._num_img_pairs)

        num_bytes += src.utils.memory.get_num_bytes(self._img_filename_codes_arr)
        num_bytes += src.utils.memory.get_num_bytes(self._img_bbox_codes_arr)

        num_bytes += src.utils.memory.get_num_bytes(self._img_anc_uid_codes_arr)
        num_bytes += src.utils.memory.get_num_bytes(self._img_pos_uid_codes_arr)
        num_bytes += src.utils.memory.get_num_bytes(self._img_neg_uid_codes_arr)

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
        print("self._neg_img_filename_list_id")
        print("  ", src.utils.pkl.pickle_test(self._neg_img_filename_list_id))

        print("self._split_num_mask")
        print("  ", src.utils.pkl.pickle_test(self._split_num_mask))

        print("self._cloth_type_list")
        print("  ", src.utils.pkl.pickle_test(self._cloth_type_list))
        print("self._cloth_subtype_llist")
        print("  ", src.utils.pkl.pickle_test(self._cloth_subtype_llist))
        print("self._cloth_type_inv_dict")
        print("  ", src.utils.pkl.pickle_test(self._cloth_type_inv_dict))
        print("self._cloth_subtype_inv_dict_list")
        print("  ", src.utils.pkl.pickle_test(self._cloth_subtype_inv_dict_list))
        
        print("self._num_imgs")
        print("  ", src.utils.pkl.pickle_test(self._num_imgs))
        print("self._num_img_pairs")
        print("  ", src.utils.pkl.pickle_test(self._num_img_pairs))
        
        print("self._img_filename_codes_arr")
        print("  ", src.utils.pkl.pickle_test(self._img_filename_codes_arr))
        print("self._img_bbox_codes_arr")
        print("  ", src.utils.pkl.pickle_test(self._img_bbox_codes_arr))

        print("self._img_anc_uid_codes_arr")
        print("  ", src.utils.pkl.pickle_test(self._img_anc_uid_codes_arr))
        print("self._img_pos_uid_codes_arr")
        print("  ", src.utils.pkl.pickle_test(self._img_pos_uid_codes_arr))
        print("self._img_neg_uid_codes_arr")
        print("  ", src.utils.pkl.pickle_test(self._img_neg_uid_codes_arr))

        print("-- PICKLE STATUS END --")



class ConsToShopClothRetrBmkImageLoader(Dataset):
    """
    Dataset class for the DeepFashion Consumer-to-shop Clothes Retrieval Benchmark dataset.
    Version for directly loading images.
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

        self._compute_cloth_type_vars()

        self._compute_num_imgs()

        self._compute_masks()

        self._store_img_filename_codes()

        self._compute_img_filename_to_img_uid_dict()

        self._store_img_bbox_codes()

        self._delete_img_filename_to_img_uid_dict()


    def __len__(self):
        
        return self._num_imgs


    def __getitem__(self, idx):
        """
        Returns a dataset item.

        :param idx: int
            Index of the dataset item.

        :return img: torch.tensor
            An image of the dataset.
        :return item_id: int
            Item ID of the image.
        """

        img_filename, item_id = self._decode_img_filename(self._get_img_filename_code(idx))

        full_img_filename = os.path.join(self._dataset_dirname, img_filename)
        img = torchvision.io.read_image(full_img_filename)
        if img.size(dim=0) == 1: img = img.repeat(3, 1, 1)

        img_bbox = self._decode_img_bbox(self._get_img_bbox_code(idx))

        x1, y1, x2, y2 = img_bbox

        img = img[:,y1:y2,x1:x2]

        if self._img_transform is not None:
            img = self._img_transform(img)

        return img, item_id
    

    def get_subset_indices(self, split=None, domain=None):
        """
        Returns dataset subset indices.

        :param split: str, optional
            Split name. Can be "train", "test", or "val".
            Split will not be taken into account if not provided.
        :param domain: str, optional
            Domain name. Can be "shop" or "consumer".
            Domain will not be taken into account if not provided.
        
        :return: np.ndarray
            An array with the split indices.
        """

        mask = np.full(shape=(self._num_imgs), fill_value=True)

        if split is not None:

            if split == "train": split_num = 0
            elif split == "val": split_num = 1
            elif split == "test": split_num = 2

            mask = np.logical_and(mask, self._split_num_mask == split_num)

        if domain is not None:

            if domain == "shop": domain_num = 0
            elif domain == "consumer": domain_num = 1

            mask = np.logical_and(mask, self._domain_num_mask == domain_num)

        idxs = np.argwhere(mask).flatten()

        return idxs


    ########
    # AUXILIARY INITIALIZATION METHODS
    ########


    def _compute_cloth_type_vars(self):
        """
        Auxiliary function called in the constructor.

        Computes cloth types and subtypes from the `img` directory and stores them into attributes.

        Cloth type attributes:
          - `_cloth_type_list`: List of the different cloth types.
          - `_cloth_type_inv_dict`: Dict that returns an idx given a cloth type.
        Cloth subtype attributes:
          - `_cloth_subtype_llist`: 2D list of the different cloth subtypes.
          - `_cloth_subtype_inv_dict_list`: List of dicts that return an idx given a cloth subtype.
        """

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
        """
        Auxiliary function called in the constructor.
        
        Computes the number of images in the dataset.
        The number of images is stored in attribute `_num_imgs`.
        The number is read from the `aux/img_filename_list.txt` file.
        """

        img_filename_list_filename = os.path.join(self._dataset_dirname, "aux", "img_filename_list.txt")
        
        img_filename_list_file = open(img_filename_list_filename, "r")

        self._num_imgs = int(img_filename_list_file.readline()[:-1])

        img_filename_list_file.close()


    def _compute_masks(self):
        """
        Auxiliary function called in the constructor.

        Computes train/test/val split and shop/consumer domain masks.
        Indices are stored in the following attributes:
          - `_split_num_mask`, where train/val/test is 0/1/2
          - `_domain_num_mask`, where shop/consumer is 0/1
        """
        
        # Shop/consumer and item ID masks

        self._domain_num_mask = np.empty(shape=(self._num_imgs), dtype=int)
        item_id_mask = np.empty(shape=(self._num_imgs), dtype=int)

        img_filename_list_filename = os.path.join(self._dataset_dirname, "aux", "img_filename_list.txt")
        img_filename_list_file = open(img_filename_list_filename, "r")

        for _ in range(2):
            img_filename_list_file.readline()

        for idx, line in enumerate(img_filename_list_file.readlines()):
            
            tkns = line.split(os.path.sep)
            domain_num = 0 if tkns[4][:-8] == "shop" else 1
            item_id = int(tkns[3][3:])

            self._domain_num_mask[idx] = domain_num
            item_id_mask[idx] = item_id

        img_filename_list_file.close()

        # Split mask

        self._split_num_mask = np.empty(shape=(self._num_imgs), dtype=int)

        item_id_set = set()

        list_eval_partition_filename = os.path.join(self._dataset_dirname, "Eval", "list_eval_partition.txt")
        list_eval_partition_file = open(list_eval_partition_filename, "r")

        for _ in range(2):
            line = list_eval_partition_file.readline()

        for line in list_eval_partition_file.readlines():

            tkns = line.split()
            item_id = int(tkns[2][3:])
            split_name = tkns[3]
            
            if item_id not in item_id_set:

                if split_name == "train": self._split_num_mask[item_id_mask == item_id] = 0
                elif split_name == "val": self._split_num_mask[item_id_mask == item_id] = 1
                elif split_name == "test": self._split_num_mask[item_id_mask == item_id] = 2

                item_id_set.add(item_id)


    def _store_img_filename_codes(self):
        """
        Auxiliary function called in the constructor.
        
        Encodes image filenames and stores them in the `_img_filename_codes_arr` attribute.
        Image filenames are read from the `aux/img_filename_list.txt` file.
        """

        self._img_filename_codes_arr = bytearray(4 * self._num_imgs)

        img_filename_list_filename = os.path.join(self._dataset_dirname, "aux", "img_filename_list.txt")
        img_filename_list_file = open(img_filename_list_filename, "r")

        for _ in range(2):
            img_filename_list_file.readline()

        for idx, line in enumerate(img_filename_list_file.readlines()):
            
            img_filename_code = self._encode_img_filename(line[:-1])          
            self._set_img_filename_code(idx, img_filename_code)

        img_filename_list_file.close()


    def _compute_img_filename_to_img_uid_dict(self):
        """
        Auxiliary function called in the constructor.
        
        Computes a dict that maps image filenames to an image uid and stores it in the
        `_img_filename_to_img_uid_dict` attribute.
        The dict serves as a temporary auxiliary variable.
        The dict is deleted afterwards in `_delete_img_filename_to_img_uid_dict`.
        """

        self._img_filename_to_img_uid_dict = {
            self._decode_img_filename(self._get_img_filename_code(idx))[0]: idx
            for idx in range(self._num_imgs)
        }


    def _store_img_bbox_codes(self):
        """
        Auxiliary function called in the constructor.
        
        Encodes image bboxes and stores them in the `_img_bbox_codes_arr` attribute.
        Image bboxes are read from the `Anno/list_bbox_consumer2shop.txt` file.
        """

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


    def _delete_img_filename_to_img_uid_dict(self):
        """
        Auxiliary function called in the constructor.
        
        Deletes the `_img_filename_to_img_uid_dict` attribute created in
        `_compute_img_filename_to_img_uid_dict`.
        """

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

        return img_filename, item_id
    

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
        num_bytes += src.utils.memory.get_num_bytes(self._domain_num_mask)

        num_bytes += src.utils.memory.get_num_bytes(self._cloth_type_list)
        num_bytes += src.utils.memory.get_num_bytes(self._cloth_subtype_llist)
        num_bytes += src.utils.memory.get_num_bytes(self._cloth_type_inv_dict)
        num_bytes += src.utils.memory.get_num_bytes(self._cloth_subtype_inv_dict_list)

        num_bytes += src.utils.memory.get_num_bytes(self._num_imgs)

        num_bytes += src.utils.memory.get_num_bytes(self._img_filename_codes_arr)
        num_bytes += src.utils.memory.get_num_bytes(self._img_bbox_codes_arr)

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
        print("self._domain_num_mask")
        print("  ", src.utils.pkl.pickle_test(self._domain_num_mask))

        print("self._cloth_type_list")
        print("  ", src.utils.pkl.pickle_test(self._cloth_type_list))
        print("self._cloth_subtype_llist")
        print("  ", src.utils.pkl.pickle_test(self._cloth_subtype_llist))
        print("self._cloth_type_inv_dict")
        print("  ", src.utils.pkl.pickle_test(self._cloth_type_inv_dict))
        print("self._cloth_subtype_inv_dict_list")
        print("  ", src.utils.pkl.pickle_test(self._cloth_subtype_inv_dict_list))
        
        print("self._num_imgs")
        print("  ", src.utils.pkl.pickle_test(self._num_imgs))
        
        print("self._img_filename_codes_arr")
        print("  ", src.utils.pkl.pickle_test(self._img_filename_codes_arr))
        print("self._img_bbox_codes_arr")
        print("  ", src.utils.pkl.pickle_test(self._img_bbox_codes_arr))

        print("-- PICKLE STATUS END --")