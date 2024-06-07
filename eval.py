## Functions for deploy & evaluation
import os
import argparse
from tqdm import tqdm

#torch libraries
import torch
import torch.nn as nn

#self defined libraies

from dataset import build_evalModel_loader
from model import TSC_M,TSC_MA
import cv2
from util import preprocess1image,predict2labels,visualize,read_image
from lanedetection import*

def evalModel(model:nn.Module,dataset_dir:str = 'GTSRB',test_all:bool=False,cropROI:bool=True):
    device = torch.device('cpu')
    model.to(device)
    
    eval_loader = build_evalModel_loader(dataset_dir=dataset_dir,batch_size=50,test_all=test_all,cropROI=cropROI)
    loss_func = nn.CrossEntropyLoss()
    
    eval_acc,eval_loss = 0.0,0.0
    #f = open('eval.log','w')
    model.eval()

    for _, data in enumerate(tqdm(eval_loader)):
        images, labels = data['images'].to(device), data['labels'].to(device) # 
        pred = model(images)
        loss = loss_func(pred, labels)

        eval_acc = eval_acc+torch.sum(torch.argmax(pred, dim=1) == labels)
        eval_loss = eval_loss+loss.item()

    print(f'total loss is {eval_loss}, dataset len is {len(eval_loader)}')
    print(f'# of accurate data  {eval_acc}, dataset len is {len(eval_loader.dataset)}')
    
def testTSR(img_paths:list[str]):
    """Test images """
    forbids = cv2.CascadeClassifier()
    warnings = cv2.CascadeClassifier()
    #suggests = cv2.CascadeClassifier()
    others = cv2.CascadeClassifier()
    forbids.load(cv2.samples.findFile(os.path.join('checkpoints','forbids.xml')))
    warnings.load(cv2.samples.findFile(os.path.join('checkpoints','warnings.xml')))
    #suggests.load(cv2.samples.findFile(os.path.join('checkpoints','suggests.xml')))
    others.load(cv2.samples.findFile(os.path.join('checkpoints','others.xml')))
    
    detectors = [forbids,warnings,others]#,suggests]
     
    ## build Traffic Sign Classifier
    device = torch.device('cpu')
    model = TSC_M(0.3)
    model.load_state_dict(torch.load(os.path.join('checkpoints','model_best.pth'), map_location=device))

    model.to(torch.device('cpu'))
    model.eval()
    
    for img_path in img_paths:
        rgbimg = cv2.imread(img_path)
        img = cv2.cvtColor(rgbimg, cv2.COLOR_BGR2GRAY)
        objects = []
        for detector in detectors:
            objects.extend(detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32)))
        
        #Traffic sign recognition
        #raise ValueError('stop')
        if len(objects) > 0:
            data = preprocess1image(img,objects).to(device)
            predict = model(data)
            traffic_signs = predict2labels(objects,predict)
            result = visualize(rgbimg,traffic_signs)
        else:
            result = rgbimg
            
        output_path = img_path.rstrip('.jpg') + '_result.jpg'
        cv2.imwrite(output_path,result)

def testLD(img_paths:list[str]):
    for img_path in img_paths:
        image = read_image(img_path) 
        binary_image = color_based_segmentation(image)   
        lines = detect_lane_lines(binary_image)  
        result_image = draw_lane_lines(image, lines)   
        cv2.imwrite('test2.jpg',result_image)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='model path', type=str, default=os.path.join('checkpoints','model_best.pth'))
    parser.add_argument('--dataset_dir', help='dataset directory', type=str, default='GTSRB')
    parser.add_argument('--drop_out_prob',help='hyper parameter for dropout layers',type=float,default=0.1)
    parser.add_argument('--bs',help='batch size',type=int,default=50)
    parser.add_argument('--noROI',action='store_true',default=False)
    parser.add_argument('--testAll',action='store_true',default=False)
    
    parser.add_argument('--testModel',action='store_true',default=False)
    parser.add_argument('--testTSR',action='store_true',default=False)
    parser.add_argument('--testLD',action='store_true',default=False)
    args = parser.parse_args()
 
    model =  TSC_M(args.drop_out_prob)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu'))
                          )
    if args.testTSR:
        testTSR([os.path.join('assets',f'test{i}.jpg') for i in range(2)])
    if args.testLD:
        testLD([os.path.join('assets','road10.jpg')])
    if args.testModel:
        evalModel(model,args.dataset_dir,args.testAll,not args.noROI)