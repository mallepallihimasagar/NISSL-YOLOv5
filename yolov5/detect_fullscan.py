import argparse
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random

import os
import cv2
import json

import numpy as np
import cv2

os.environ['OPENCV_IO_ENABLE_JASPER'] = '1'

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, help='model.pt path(s)',required=True)
parser.add_argument('--jp2_section_path',type=str,required=True, help='path to the nissl jp2 section')
parser.add_argument('--output_filename',type=str,default='section.png',help='filename of output scan')
# parser.add_argument('--source', type=str, default='../test/images', help='source')  # file/folder, 0 for webcam
# parser.add_argument('--img-size', type=int, default=500, help='inference size (pixels)')
# parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
# parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
# parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# parser.add_argument('--view-img', action='store_true', help='display results')
# parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
# parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
# parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
# parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
# parser.add_argument('--augment', action='store_true', help='augmented inference')
# parser.add_argument('--update', action='store_true', help='update all models')
# parser.add_argument('--project', default='runs/detect', help='save results to project/name')
# parser.add_argument('--name', default='exp', help='save results to project/name')
# parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
# opt = parser.parse_args()
# print(opt)
# check_requirements()
args = parser.parse_args()
from dotmap import DotMap
# opt = {'weights': ['runs/train/yolov5s_results/weights/best.pt'], 'source': '../point_cloud', 'img_size': 500, 'conf_thres': 0.4, 'iou_thres': 0.45, 'device': '', 'view_img': False, 'save_txt': False, 'save_conf': False, 'classes': None, 'agnostic_nms': False, 'augment': False, 'update': False, 'project': 'runs/detect', 'name': 'exp', 'exist_ok': False}


# opt = DotMap(opt)



def detect(weights = 'runs/train/yolov5s_results/weights/best.pt',save_img=False):
    opt = {'weights': weights, 'source': 'point_cloud', 'img_size': 500, 'conf_thres': 0.4, 'iou_thres': 0.45, 'device': '', 'view_img': False, 'save_txt': False, 'save_conf': False, 'classes': None, 'agnostic_nms': False, 'augment': False, 'update': False, 'project': 'runs/detect', 'name': 'exp', 'exist_ok': False}


    opt = DotMap(opt)
    preds = []
    total_time = 0
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        start = time.time()
        pred = model(img, augment=opt.augment)[0]
        total_time +=(time.time()-start)

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        
        return pred


filename = args.jp2_section_path #'/content/Hua167&166-N24-2014.02.20-10.42.43_Hua167_1_0069.jp2'
weights = args.weights
output_filename = args.output_filename
img  = cv2.imread(filename)
img_uint8 = np.uint8(img)

img8 = img_uint8.copy()

#starting index are ci, ri
ci, ri = 0,0
x ,y = ci*500, ri*500

grid_size_x = 36
grid_size_y = 48
tile_size = 500
#crop the region from the section
region = img8[x:x+(tile_size*grid_size_x),y:y+(tile_size*grid_size_y)]

#verify the region size
assert tile_size*grid_size_x == region.shape[0]
assert tile_size*grid_size_y == region.shape[1]

try:
    os.mkdir('point_cloud')
except:
    pass

colors = [(0,0,255),(0,255,0),(255,0,0)]
normalize = 500/512
coordinates =[]
#now for each tile in grid, run the forward pass and get the centers
for j in range(grid_size_y):
    for i in range(grid_size_x):
        tile = region[i*500:(i+1)*500, j*500:(j+1)*500]

        # do preprocessing and send it to model get the coordiantes
        #save the tile in jpg to /content/point_cloud
        cv2.imwrite('point_cloud/tile.jpg',tile)
        with torch.no_grad():
            preds = detect(weights)
            for ii, det in enumerate(preds):
                det = det.cpu()
                for *xyxy, conf, cls in reversed(det):
                    x1,y1,x2,y2 = xyxy[0].item(),xyxy[1].item(),xyxy[2].item(),xyxy[3].item()
                    # normalize = 500/512
                    x1 = int(x1*normalize)
                    y1 = int(y1*normalize)
                    x2 = int(x2*normalize)
                    y2 = int(y2*normalize)

                    center_x = int(x1 + (x2-x1)/2)
                    center_y = int(y1 + (y2-y1)/2)

                    #boxed_tile = cv2.rectangle(new_tile,(x1,y1),(x2,y2),colors[int(cls.item())],thickness=2)
                    coordinates.append({"X":center_x+j*500,"Y":center_y+i*500,"cls":int(cls.item())})
                    cv2.circle(tile,(center_x,center_y),3,colors[int(cls.item())],-1)
                #cv2_imshow(tile)
        print("\n",i,j)

cv2.imwrite(output_filename,region)
scaled_region = cv2.resize(region,(9600,7200))
cv2.imwrite(output_filename[:-4]+'_scaled.png',scaled_region)

print(f'Output on full scan saved to {output_filename}')
print(f'Resized Output on full scan saved to {output_filename[:-4]+"_scaled.png"}')

import json
with open(output_filename[:-4]+'_centers.json','w') as f:
    json.dump(coordinates,f)
print(f'Output coordinates on full scan saved to {output_filename[:-4]+"_centers.json"}')







