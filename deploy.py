## Functions for deploy & evaluation
import os
import argparse
from tqdm import tqdm
import cv2 

#torch libraries
import torch
import torch.nn as nn

#self defined libraies

from dataset import build_eval_loader
from model import TSC_M,TSC_MA


def main():
    ## build Traffic Sign Detector
    forbids = cv2.CascadeClassifier()
    warnings = cv2.CascadeClassifier()
    suggests = cv2.CascadeClassifier()
    others = cv2.CascadeClassifier()
    forbids.load(cv2.samples.findFile(os.path.join('checkpoints','forbids.xml')))
    warnings.load(cv2.samples.findFile(os.path.join('checkpoints','warnings.xml')))
    suggests.load(cv2.samples.findFile(os.path.join('checkpoints','suggests.xml')))
    others.load(cv2.samples.findFile(os.path.join('checkpoints','others.xml')))    
    ## build Traffic Sign Classifier
    model = TSC_M(0.3).to(torch.device('cpu'))
    model.load_state_dict(os.path.join('checkpoints','model_best.pth'))
    ## build lane detetion module
    # TODO: Tiana ^^
    
    ## inference from camera  
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascades/haarcascade_frontalface_alt.xml')
    parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
    args = parser.parse_args()
    face_cascade_name = args.face_cascade
    eyes_cascade_name = args.eyes_cascade
    face_cascade = cv2.CascadeClassifier()
    eyes_cascade = cv2.CascadeClassifier()
    
    #-- 1. Load the cascades
    if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)
    camera_device = args.camera
    #-- 2. Read the video stream
    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
            detectAndDisplay(frame)
        if cv2.waitKey(10) == 27:
            break