# FCOS
there are many files,but only a few of them will by useful for your training and evaluation.  
train_voc.py: run directly for training  
predict.py: run directly to use model to predict your sample  
eval_voc.py: run directly to get mAP on Voc testset of your model  
checkpoint file: to save model trained from train_voc.py  

before your train you should download this model:
resnet50：https://download.pytorch.org/models/resnet50-0676ba61.pth
put it at the same dirpath as train_voc.py
