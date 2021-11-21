# FishNet


This repo holds the soruce code for CE7454: Deeplearning group assignment work. A deep dive into fishnet with deep fish data and cifar. Please refer to the report for further details of our implementation, contribution and findings.

The source code utilizes the original FishNet model published here https://github.com/kevin-ssy/FishNet as well as the DeepFish Dataset, whose code is published here https://github.com/alzayats/DeepFish

This project applies the FishNet Model on the DeepFish dataset and shows that on the counting and classification tasks, we can achieve better performance than that of the ResNet50 model reported in the DeepFish paper.The DeepFish code has been cloned and reworked quite extensively for this project, the code from the original source will not run as intended here.

The model classes in this repository will load the FishNet150 model from the original FishNet code as a module. It will also make a reference to kevin-ssy's pre-trained FishNet150 model, and requires the checkpoint file for the model pre-trained without tricks https://www.dropbox.com/s/hjadcef18ln3o2v/fishnet150_ckpt.tar?dl=0 If hyperlink doesa not work. Please refer back to the author's original github

### Prerequisites
- Python 3.6.x
- PyTorch 0.4.0+

### Training

To get started, simply run main.py
```
python main.py --task <The task>
```
There are serveral tasks that can be done with Fishnet architecture.
For segmentation tasks on the deepfish dataset simply run.
```
python main.py --task segmentation
```
or 

```
python main.py --task ablation
```
to run the ablation study code on cifar 100 dataset.

The list of tasks includes: 
<li>
ablation
classification
counting
segmentation
</li>



### Citation
[FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction](http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf)
, Shuyang Sun, Jiangmiao Pang, Jianping Shi, Shuai Yi, Wanli Ouyang, NeurIPS 2018.
FishNet was used as a key component
 for winning the 1st place in [COCO Detection Challenge 2018](http://cocodataset.org/#detection-leaderboard).

Our work is also heavily influenced by the paper:
```
@inproceedings{sun2018fishnet,
  title={FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction},
  author={Sun, Shuyang and Pang, Jiangmiao and Shi, Jianping and Yi, Shuai and Ouyang, Wanli},
  booktitle={Advances in Neural Information Processing Systems},
  pages={760--770},
  year={2018}
}
```
