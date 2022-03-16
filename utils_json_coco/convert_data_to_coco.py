# import functions
from labelme2coco import get_coco_from_labelme_folder, save_json

# set labelme training data directory
labelme_train_folder = "/home/josyula/Documents/DataAndModels/pruning_training/stacked/train"

# set labelme validation data directory
# labelme_val_folder = "/home/josyula/Documents/DataAndModels/pruning_training/val"
#
# # set labelme test data directory
# labelme_test_folder = "/home/josyula/Documents/DataAndModels/pruning_training/test"

# set path for coco json to be saved
export_dir = "/home/josyula/Documents/DataAndModels/pruning_training/stacked/train/labels"

# create train coco object
train_coco = get_coco_from_labelme_folder(labelme_train_folder)

# export train coco json
save_json(train_coco.json, export_dir+"train.json")
#
# # create val coco object
# val_coco = get_coco_from_labelme_folder(labelme_val_folder, coco_category_list=train_coco.json_categories)
#
# # export val coco json
# save_json(val_coco.json, export_dir+"val.json")
#
# # create test coco object
# test_coco = get_coco_from_labelme_folder(labelme_test_folder, coco_category_list=train_coco.json_categories)
#
# # export val coco json
# save_json(test_coco.json, export_dir+"test.json")