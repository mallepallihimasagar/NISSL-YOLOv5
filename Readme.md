# NISSL PROJECT
Nissl cell detection using [YOLOv5](https://github.com/ultralytics/yolov5) 

## Installation

The code uses pytorch framework torch >= 1.8 and cuda >= 10.0.

Clone the repo and execute to following commands for training and testing
### Cloning the repo
```bash
git clone https://github.com/mallepallihimasagar/NISSL-YOLOv5.git
cd NISSL-YOLOv5/yolov5

#Install Dependencies
pip install -r requirements.txt
```
For all the executions current path should be ** ~/NISSL-YOLOv5/yolov5/ **

Trained weights file (.pt file) ** ~/NISSL-YOLOv5/yolov5/trained_weights/best.pt **
### Training the model
The Ground Truth Data are in Yolo format, no conversion is needed.
 
```bash
!python train.py --img <img size> --batch <batch size> --epochs <num epochs> --data '../data.yaml' --cfg ./models/custom_yolov5s.yaml --weights <pretrained weights> --name <output folder>

## Example
!python train.py --img 500 --batch 1 --epochs 1 --data '../data.yaml' --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results
```
### Testing
```bash
python detect.py --weights trained_weights/best.pt --img 500 --conf 0.4 --iou-thres 0.45 --source ../test/images
```
### Testing on Full Section (18k x 24k) 
Add the 16-bit Jp2 files to ** ~/NISSL-YOLOv5/yolov5/jp2_section/ **  folder without normalization.
```bash
python detect_fullscan.py --weights <trained weights.pt> --jp2_section_path <path to jp2 file> --output_filename <output file_name.png>

#Example
python detect_fullscan.py --weights trained_weights/best.pt --jp2_section_path jp2_section/Hua167%26166-N24-2014.02.20-10.42.43_Hua167_1_0070.jp2 --output_filename section70.png
```

## Credits 
#### [Ultralytics](https://ultralytics.com/)

Original release [Ultralytics/yolov5](https://github.com/ultralytics/yolov5)