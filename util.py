import os
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as F

##### For Training Classifier #####
def write_result_log(logfile_path, epoch, epoch_time, train_acc, val_acc, train_loss, val_loss, is_better,epochs):
    ''' write experiment log file for result of each epoch to ./experiment/{exp_name}/log/result_log.txt '''
    with open(logfile_path, 'a') as f:
        f.write(f'[{epoch + 1}/{epochs}] {epoch_time:.2f} sec(s) Train Acc: {train_acc:.5f} | Val Acc: {val_acc:.5f} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}')
        if is_better:
            f.write(' -> val best (acc)')
        f.write('\n')
        
def plot_learning_curve(logfile_dir, result_lists):

    plot_list = ['train_acc','train_loss','val_acc','val_loss']
    for plot_name in plot_list:
        data_list = result_lists[plot_name]
        epochs = range(len(data_list))
        plt.plot(epochs, data_list, marker='o', linestyle='-')
        plt.title('Learning Curve Of'+plot_name)
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.savefig(os.path.join(logfile_dir,plot_name+'.png'))
        plt.clf( )

def read_image(img_path:str):
    # read image
    if img_path.endswith('road10.jpg'):
        image = cv2.imread(img_path)
        crop_img = image[300:300+1000, 0:0+3000]
    else:
        image = cv2.imread(img_path)
        h,w,_ = image.shape
        crop_img = image[h//2:h, :]
        
    return crop_img

### For Deploying, After detecting ROI, main will call this function for classifier
def preprocess1image(img:np.ndarray,objects:list[tuple[int]])->torch.Tensor:
    """img is gray image, crop ROI one by one according to (x,y,w,h) in objects"""
    ROI = None
    for (x, y, w, h) in objects:
        ### image preprocess
        crop_img = img[y:y+h, x:x+w]
        crop_img = cv2.resize(crop_img, (32, 32))
        crop_img = cv2.equalizeHist(crop_img)
        ### turn to tensor
        data = torch.tensor(crop_img/255.0).unsqueeze(dim=0).to(torch.float32)
        data = F.normalize(data,(0.5,),(0.5,))
        if ROI is None:
            ROI = data
        else:
            ROI = torch.cat((ROI,data),dim=0)
    
    if len(ROI.shape) == 3:
        ROI = ROI.unsqueeze(dim=0)
    assert len(ROI.shape) == 4,'data shape could be error'
    return ROI

##### For Visualization
def predict2labels(objects:list[tuple[int]],predict:torch.Tensor,threshold = 0.7)->list[tuple[int]]:
    conf,classes = torch.max(predict,dim=1)
    labels = []
    for idx,(x,y,w,h) in enumerate(objects):
        if conf[idx] > threshold: #threshold that can be adjusted
            labels.append((x,y,w,h,classes[idx].item()))
    
    return labels

def visualize(rgbimg:np.ndarray,signs:list[tuple[int]])->np.ndarray:
    #Class label 
    forbids = [0,1,2,3,4,5,7,8,9,10,15,16] + [6,17,32,41,42] #red
    warnings = [11,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    suggests = [33,34,35,36,37,38,39,40]
    others = [12,13,14]
    colors = {'forbids':(0,0,255),'warnings':(0,255,255),'suggests':(255,0,0),'others':(0,0,0 ) } #red, yellow,blue,black
    #draw signs and roads on the given BGR image
    class_name = ['speed lim 20','speed lim 30','speed lim 50','speed lim 60','speed lim 70','speed lim 80','no 80','speed lim 100','speed lim 120','no pass',
                  'no cross','baby','renovation','give way','stop','no car','no truck','no entry','danger','left',
                  'right','l-r curve','uneven','wind','right shrink','construction','traffic light','pedestrian','childs','bikes',
                  'frozen','animals','forbid','turn right','turn left','go front','front right','front left','right lane','left lane',
                  'roundabout','no pass','no cross']
    # text parameters
    thickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    img = np.array(rgbimg)
    for (x,y,w,h,class_id) in signs:
        
        if class_id in forbids:
            color = colors['forbids']
        elif class_id in warnings:
            color = colors['warnings']
        elif class_id in suggests:
            color = colors['suggests']
        else:
            color = colors['others']
            
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)

        # Define the text to add
        text = class_name[class_id]

        text_color = (255, 255, 255)  # White color

        # Get the size of the text box
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

        # Set the text start position
        text_x = x
        text_y = y - 10  # Position text above the bounding box

    #    Draw a filled rectangle as background for text for better visibility
        cv2.rectangle(img, (text_x, text_y - text_height - baseline), 
                (text_x + text_width, text_y + baseline), 
                color, cv2.FILLED)

        # Add the text on the image
        cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
    
    return  img