## To Train for Prunning Point Detection
Navigate to prunning_train and run custom_trainv1-4.py depending on your requirements. 
- 1 RGB, MaskRCNN Instance Segmentation
- 2 Augmented Data, Rotation + sheer + noise, FastRCNN, Object Detection
- 3 Mask+opticalflow+depth,  MaskRCNN Instance Segmentation
- 4 Stacked Mask,Opticalflow,Depth, image size (:,:,4)  MaskRCNN Instance Segmentation
-  
```
python custom_train.py
```
Video demonstration of Pruning 


![Olson Farm Pruning Demo](https://user-images.githubusercontent.com/2005601/226229550-18693208-6a57-48f7-a3e8-9f4e2cc2bc7f.mp4)



The output of RGB segmentation 

![pruning_labeled_data](output_old/3pred.jpg)

The output using the mask+opticalflow+depth model 

![pruning_labeled_data](output_old/combinedResult2.jpg)

The output using the Augmented Data 

![pruning_labeled_data](output_old/result.jpg)


## About Dectectron2

<img src=".github/Detectron2-Logo-Horz.svg" width="300" >


Detectron2 is Facebook AI Research's next generation library
that provides state-of-the-art detection and segmentation algorithms.
It is the successor of
[Detectron](https://github.com/facebookresearch/Detectron/)
and [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).
It supports a number of computer vision research projects and production applications in Facebook.


## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron2

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
