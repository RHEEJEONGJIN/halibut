from pathlib import Path
from typing import List

import sys
import numpy as np
from PIL import Image
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import list_files_recursively, load_json, save_json
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class labelme2coco:
    def __init__(self):
        raise RuntimeError(
            "Use labelme2coco.convert() or labelme2coco.get_coco_from_labelme_folder() instead."
        )


def get_coco(
    labelme_folder: str = '', export_dir: str = '', to_replace: str = '', from_replace: str = ''
) -> Coco:
    # get json list

    _, abs_json_path_list = list_files_recursively(labelme_folder, contains=[".json"])
    labelme_json_list = abs_json_path_list
    print(len(labelme_json_list))

    train_list, val_list = train_test_split(labelme_json_list, test_size=0.2, random_state=42)

    # create train coco
    train_coco = create_coco(train_list)
    save_json(train_coco.json, export_dir+"/train.json")

    # create val coco
    val_coco = create_coco(val_list, coco_category_list=train_coco.json_categories)
    save_json(val_coco.json, export_dir+"/val.json")


def create_coco(labelme_json_list, coco_category_list=None):
    # init coco object
    coco = Coco()

    if coco_category_list is not None:
        coco.add_categories_from_coco_category_list(coco_category_list)

    # parse labelme annotations
    category_ind = 0
    for json_path in tqdm(
        labelme_json_list, "Converting labelme annotations to COCO format"
    ):
        data = load_json(json_path)
        # get image size
        image_path = json_path.replace('labels', 'images').replace('.json', '.jpg')
        width, height = Image.open(image_path).size
        imagePath = image_path.replace(to_replace, from_replace)
        # init coco image
        coco_image = CocoImage(file_name=imagePath, height=height, width=width)
        # iterate over annotations
        for shape in data["shapes"]:
            # set category name and id
            category_name = shape["label"]
            category_id = None
            for (
                coco_category_id,
                coco_category_name,
            ) in coco.category_mapping.items():
                if category_name == coco_category_name:
                    category_id = coco_category_id
                    break
            # add category if not present
            if category_id is None:
                category_id = category_ind
                coco.add_category(CocoCategory(id=category_id, name=category_name))
                category_ind += 1
            # parse bbox/segmentation
            if shape["shape_type"] == "rectangle":
                x1 = shape["points"][0][0]
                y1 = shape["points"][0][1]
                x2 = shape["points"][1][0]
                y2 = shape["points"][1][1]
                coco_annotation = CocoAnnotation(
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    category_id=category_id,
                    category_name=category_name,
                )
            elif shape["shape_type"] == "polygon":
                segmentation = [np.asarray(shape["points"]).flatten().tolist()]
                coco_annotation = CocoAnnotation(
                    segmentation=segmentation,
                    category_id=category_id,
                    category_name=category_name,
                )
            else:
                raise NotImplementedError(
                    f'shape_type={shape["shape_type"]} not supported.'
                )
            coco_image.add_annotation(coco_annotation)
        coco.add_image(coco_image)
    return coco


if __name__ == "__main__":
    labelme_folder = 'datasets/silryuk/dataset/TOD_dataset/data'
    export_dir = 'datasets/silryuk/dataset/TOD_dataset/annotations'
    to_replace = 'datasets/silryuk/dataset/TOD_dataset'
    from_replace = '../datasets/silryuk/dataset/TOD_dataset'

    coco = get_coco(
        labelme_folder=labelme_folder,
        export_dir=export_dir,
        to_replace=to_replace,
        from_replace=from_replace
        )
