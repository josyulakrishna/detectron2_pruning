import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import list_files_recursively, load_json, save_json
from tqdm import tqdm
import os.path
import glob
import pdb

def get_coco_from_labelme_folder(
    labelme_folder: str, coco_category_list: List = None
) -> Coco:
    """
    Args:
        labelme_folder: folder that contains labelme annotations and image files
        coco_category_list: start from a predefined coco cateory list
    # get json list
    """
    _, abs_json_path_list = list_files_recursively(labelme_folder, contains=[".json"])
    labelme_json_list = abs_json_path_list
    category_ind = 0

    for json_path in tqdm(labelme_json_list, "Converting labelme annotations to COCO format"):
        # print(json_path)
        data = load_json(json_path)
        image_path = str(Path(labelme_folder) / data["imagePath"])

        nameOrg = json_path.split("/")[-1].split(".")[0]+".png"
        data["imagePath"] = nameOrg
        data_json = json.dumps(data)

        # pdb.set_trace()

        with open(json_path, 'w') as outfile:
            # json.dump(data_json, outfile)
            outfile.write(data_json)
            print("written Orig")

        flowPath = json_path.split("/")[:-1]
        nameFlow = json_path.split("/")[-1].split(".")[0]+"_flow.png"
        nameFlowJSON = json_path.split("/")[-1].split(".")[0]+"_flow.json"
        flowPath.append(nameFlowJSON)
        flowPath = '/'.join(flowPath)

        data["imagePath"] = nameFlow
        data_json = json.dumps(data)

        with open(flowPath, 'w') as outfile:
            # json.dump(data_json, outfile)
            outfile.write(data_json)
            print("written flow")


        maskPath = json_path.split("/")[:-1]
        namemask = json_path.split("/")[-1].split(".")[0]+"_mask.png"
        namemaskJSON = json_path.split("/")[-1].split(".")[0]+"_mask.json"
        maskPath.append(namemaskJSON)
        maskPath = '/'.join(maskPath)

        data["imagePath"] = namemask
        data_json = json.dumps(data)

        with open(maskPath, 'w') as outfile:
            # json.dump(data_json, outfile)
            outfile.write(data_json)
            print("written mask")

if __name__=="__main__":
    # get_coco_from_labelme_folder("~/Documents/DataAndModels/labeled_data")
    get_coco_from_labelme_folder("/home/josyula/Documents/DataAndModels/pruning_training/detectron_training")