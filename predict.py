import torch
import numpy as np
from model.fcos import FCOSDetector
from PIL import Image
from torchvision import transforms
import json
import os
from draw_box_utils import draw_objs
import matplotlib.pyplot as plt

# hyper par
box_thresh=0.5 # 阈值一般设为0.5不变
line_thickness=3
font_size=20
model_name = '30'
img_dir = "C:/Users/86153/Desktop/hw2_v2/test/car_behind_bike.jpg"
img_dir = "D:/历史照片/这两年/psc.jfif"
img_dir = "D:/手机存档/照片/收藏/球球.jpg"
img_dir = r'C:\Users\86153\Desktop\hw2_v2\voctest\VOCdevkit\VOC2007\JPEGImages\000001.jpg'

# read class_indict
label_json_path = './pascal_voc_classes.json'
assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
with open(label_json_path, 'r') as f:
    class_dict = json.load(f)

category_index = {str(v): str(k) for k, v in class_dict.items()}

model=FCOSDetector(mode="inference")
model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load("./checkpoint/model_{}.pth".format(model_name),map_location=torch.device('cpu')))

model=model.cuda().eval()
# model.eval()

# original_img = Image.open("D:/历史照片/这两年/psc.jfif") # 0 364 857 2134 # 864 486 1600 2134
original_img = Image.open(img_dir) # 0 143 1665 1158
# original_img = Image.open("test/car_behind_bike.jpg") # 141 247 1426 1032 # 476 345 801 644

original_img = original_img.convert('RGB') # 即使是png图片四通道也转成普通三通道

# from pil image to tensor, do not normalize image
data_transform = transforms.Compose([transforms.ToTensor()])
img = data_transform(original_img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

with torch.no_grad():
    predictions = model(img)
    predict_scores, predict_classes,predict_boxes = predictions[0][0],predictions[1][0],predictions[2][0]
    predict_scores = predict_scores.cpu().numpy()
    predict_classes = predict_classes.cpu().numpy()
    predict_boxes = predict_boxes.cpu().numpy()

    plot_img = draw_objs(original_img,
                         predict_boxes,
                         predict_classes,
                         predict_scores,
                         category_index=category_index,
                         box_thresh=box_thresh,
                         line_thickness=line_thickness,
                         font='arial.ttf',
                         font_size=font_size)


    plt.imshow(plot_img)
    plt.show()


    # 保存预测的图片结果
    # plot_img.save("test_result.jpg")

    def count_Iou(l1, l2):
        x1, y1, x2, y2 = l1
        xg1, yg1, xg2, yg2 = l2
        s1 = (x2 - x1) * (y2 - y1)
        s2 = (xg1 - xg2) * (yg1 - yg2)
        if xg1 > x2 or x1 > xg2 or yg1 > y2 or y1 > yg2:
            return 0
        xi = min(x2, xg2) - max(x1, xg1)
        yi = min(y2, yg2) - max(y1, yg1)
        s_intersec = xi * yi
        return s_intersec / (s1 + s2 - s_intersec)

    # count_Iou(predict_boxes[0], [0 ,143 ,1665 ,1158])