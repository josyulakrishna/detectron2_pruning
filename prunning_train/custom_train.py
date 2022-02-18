from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
from detectron2 import model_zoo
import cv2
import os
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import  Visualizer
import random
from detectron2.engine import DefaultPredictor

######## REGISTER DATASET #################
print("registering")
register_coco_instances("pruningTrain", {},"/home/josyula/Documents/DataAndModels/labeled_data/train/runs/labelme2coco/train.json", "/home/josyula/Documents/DataAndModels/labeled_data/train")
register_coco_instances("pruningVal", {}, "/home/josyula/Documents/DataAndModels/labeled_data/val/runs/labelme2coco/val.json", "/home/josyula/Documents/DataAndModels/labeled_data/val")
register_coco_instances("pruningTest", {}, "/home/josyula/Documents/DataAndModels/labeled_data/test/runs/labelme2coco/test.json", "/home/josyula/Documents/DataAndModels/labeled_data/test")
# dataset_dicts = DatasetCatalog.get("prunningTrain")

for data_ in ["pruningTrain", "pruningVal", "pruningTest"]:
    dataset_dicts = DatasetCatalog.get(data_)
    pruning_meta_data = MetadataCatalog.get(data_).thing_classes
    print(pruning_meta_data)
###########################################

################## VISUALIZE ##############
# import random
# from detectron2.utils.visualizer import Visualizer
#
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=pruning_meta_data, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow("1", vis.get_image()[:, :, ::-1])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
###########################################


############ TRAINING #####################
#
# class CocoTrainer(DefaultTrainer):
#
#   @classmethod
#   def build_evaluator(cls, cfg, dataset_name, output_folder=None):
#
#     if output_folder is None:
#         os.makedirs("coco_eval", exist_ok=True)
#         output_folder = "coco_eval"
#
#     return COCOEvaluator(dataset_name, cfg, False, output_folder)
#
#
# # select from modelzoo here: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md#coco-object-detection-baselines
#
#
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
# cfg.DATASETS.TRAIN = ("pruningTrain",)
# cfg.DATASETS.TEST = ("pruningVal",)
#
#
# cfg.DATALOADER.NUM_WORKERS = 4
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.SOLVER.IMS_PER_BATCH = 4
# cfg.SOLVER.BASE_LR = 0.001
#
#
# cfg.SOLVER.WARMUP_ITERS = 1000
# cfg.SOLVER.MAX_ITER = 5000 #adjust up if val mAP is still rising, adjust down if overfit
# cfg.SOLVER.STEPS = (1000, 1500)
# cfg.SOLVER.GAMMA = 0.05
#
#
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
# # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6 #your number of classes + 1
#
# cfg.TEST.EVAL_PERIOD = 500
#
# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = CocoTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()

###########################################

######### PREDICTION #####################
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.WEIGHTS = os.path.join("/home/josyula/Programs/detectron2/output/model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
# cfg.DATASETS.TEST = ("pruningTest", )
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 #your number of classes + 1
dataset_dicts = DatasetCatalog.get("pruningTest")
pruning_meta_data = MetadataCatalog.get("pruningVal")
predictor = DefaultPredictor(cfg)
# random.sample(dataset_dicts, 3)
i=0
for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    print(d["file_name"])
    outputs = predictor(im)
    # print(outputs)
    v = Visualizer(im[:, :, ::-1],
                   metadata=pruning_meta_data,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imwrite("/home/josyula/Documents/DataAndModels/labeled_data/test/"+str(i)+"pred.jpg", v.get_image()[:, :, ::-1])
    i+=1
    cv2.imshow("1", v.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
##################
# pred_classes = output_dict['instances'].pred_classes.cpu().tolist()
# class_names = MetadataCatalog.get("mydataset").thing_classes
# pred_class_names = list(map(lambda x: class_names[x], pred_classes))
########