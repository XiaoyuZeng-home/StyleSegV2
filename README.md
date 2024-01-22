# StyleSegV2
This is the implementation of the StyleSeg V2: **Towards Robust One-shot Segmentation of Brain Tissue via Optimization-free Registration Error Perception** 
## Install
The packages and their corresponding version we used in this repository are listed in below:
- Tensorflow==1.15.4
* Keras==2.3.1
+ tflearn==0.5.0
## Training
After configuring the environment, please use this command to train the model sequentially:
### Unsupervised registration training
Please train a unsupervised registration model (reg-model) to initialize the iteration. The command is:  
```
python train.py --reg_lr 1e-4  -d ./dataset/OASIS.json -c weights/xxx --clear_steps -g 0 --reg_round 2000 --scheme reg
```
### Semi-supervised segmentation and weakly supervised registration iteration
With the pretrained-model, please train a semi-supervised segmentation model (seg-model) with the reg-model fixed fistly, then fix the seg-model to train a weakly-supervised reg-model, triggering a new round of iteration. The commands of such two process are as below:
```
python train.py --seg_lr 1e-3  -d ./dataset/OASIS.json -c weights/xxx --clear_steps -g 0 --seg_round 1000 --scheme seg #Semi-supervised segmentation
python train.py --reg_lr 1e-4  -d ./dataset/OASIS.json -c weights/xxx --clear_steps -g 0 --reg_round 2000 --scheme reg_supervise #Weakly-supervised registration
```
### Iterative registration and segmentation training 
Besides, StyleSeg can be directly trainied iteratively use the following command:
```
python train.py --reg_lr 1e-4  --seg_lr 1e-3  -d ./dataset/OASIS.json --clear_steps -g 0 --reg_round 2000 --seg_round 1000 --iter_num 3 #Iterative registration and segmentation training
```
## Testing
To predict the final segmentation results in test set, use the command below:
```
python predict.py -c weights/xxx -d ./datasets/OASIS.json -g 0 --scheme seg
```
## Acknowledgment
Some codes are modified from [RCN](https://github.com/microsoft/Recursive-Cascaded-Networks) and [StyleSeg](https://github.com/JinxLv/StyleSegv1-One-shot-image-segmentation). Thanks a lot for their great contributions.
