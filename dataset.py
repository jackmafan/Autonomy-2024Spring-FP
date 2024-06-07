### Dateset functions for detector & classfiers
import os
import matplotlib.pyplot as plt
import csv
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image,ImageOps
import random

#### 
# Following are the datasets we used to train our detector and classifier
# The German Traffic Sign Recognition Benchmark
# OpenStreetView-5M

# These functions are used to extract positive and negative labels




# These functions are used to extract GTSRB to train a cnn-based classifier

def collect_metadata(data_set_dir:str):
    img_paths = [] # images
    labels = [] # corresponding labels
    ROIs = [] #region of interest, in the form of(int,int,intint)
    img_sizes = []
    for _ in range(43): ## fix the bug that all sublists point to the same list object
        img_paths.append(list())
        labels.append(list())
        ROIs.append(list())
        img_sizes.append(list())
    for c in range(0,43): 
        ### 43 classes in total
        ### all images of same class is put in same directory
        prefix = os.path.join(data_set_dir,format(c, '05d'))
        gtFile_name = 'GT-'+ format(c, '05d') + '.csv'
        gtFile = open(os.path.join(prefix, gtFile_name) )  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        
        # skip header
        # loop over all images in current annotations file
        for i,row in enumerate(gtReader):
            if i !=0:
                img_paths[c].append( os.path.join(prefix,row[0])  ) # the 1th column is the filename
                img_sizes[c].append((int(row[1]),int(row[2]))) #tuple of (w,h)
                ROIs[c].append((int(row[3]),int(row[4]),int(row[5]),int(row[6]))) #tuple of (x1,y1,x2,y2)
                labels[c].append(int(row[7])) # the 8th column is the label
        gtFile.close()
    return img_paths,labels,ROIs,img_sizes

def build_loader(data_set_dir:str,batch_size:int,cropROI:bool)->tuple[DataLoader,DataLoader]:    
    #Collect all metadata
    img_paths,labels,ROIs,img_sizes =  collect_metadata(data_set_dir)

    val_paths,val_labels,val_ROIs,val_sizes = [],[],[],[]
    train_paths,train_labels,train_ROIs,train_sizes = [],[],[],[]
    
    #Split them to train set and val set
    random.seed(0)
    for id_ in range(0,43):
        class_len = len(img_paths[id_])
        val_len = int(0.3*class_len)
        val_subset = random.sample(range(0,class_len),val_len)
        for idx in range(class_len):
            if idx in val_subset:
                val_paths.append(img_paths[id_][idx])
                val_labels.append(labels[id_][idx])
                val_ROIs.append(ROIs[id_][idx])
                val_sizes.append(img_sizes[id_][idx])
            else:
                train_paths.append(img_paths[id_][idx])
                train_labels.append(labels[id_][idx])
                train_ROIs.append(ROIs[id_][idx])
                train_sizes.append(img_sizes[id_][idx])
    
    
    val_set = GTRSB(paths = val_paths, labels = val_labels,
                      ROI = val_ROIs, sizes = val_sizes, cropROI = cropROI)
    train_set = GTRSB(paths = train_paths, labels = train_labels,
                      ROI = train_ROIs, sizes = train_sizes, cropROI=cropROI)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    return [train_loader,val_loader]

def build_evalModel_loader(dataset_dir:str,batch_size:int,test_all:bool,cropROI:bool)->DataLoader:
    #Collect all metadata
    img_paths,labels,ROIs,img_sizes =  collect_metadata(dataset_dir)

    val_paths,val_labels,val_ROIs,val_sizes = [],[],[],[]
    
    random.seed(0)
    for id_ in range(0,43):
        class_len = len(img_paths[id_])
        val_len = int(0.3*class_len)
        val_subset = random.sample(range(0,class_len),val_len)
        for idx in range(class_len):
            if idx in val_subset or test_all:
                val_paths.append(img_paths[id_][idx])
                val_labels.append(labels[id_][idx])
                val_ROIs.append(ROIs[id_][idx])
                val_sizes.append(img_sizes[id_][idx])
            # else not join
    val_set = GTRSB(paths = val_paths, labels = val_labels,
                      ROI = val_ROIs, sizes = val_sizes, cropROI = cropROI)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    
    return val_loader

class GTRSB(Dataset):
    def __init__(self,paths:list,labels:list,ROI:list,sizes:list,cropROI:bool) -> None:
        super(GTRSB,self).__init__()
        self.img_paths = paths
        self.labels = labels
        self.ROI = ROI
        self.img_size = sizes
        self.cropROI=cropROI
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,))
        ])
    def  __len__(self):
        return self.img_paths.__len__()
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = self.preprocess(img_path,index)
        img = self.transform(img)
        return {'images': img,'labels':self.labels[index] } ###seems that labels accept the form of single integer
    def preprocess(self,img_path,index):
        ### TODO-Preprocess
        ### 1: RGB to gray
        ### 2: equalize hist
        ### 3: Norm
        ### 4: Reshape to (32,32)
        rgb_img = Image.open(img_path)
        gray_img = ImageOps.grayscale(rgb_img)
        if self.cropROI:
            x1,y1,x2,y2 = self.ROI[index]
            gray_img = gray_img.crop((x1,y1,x2,y2))
        return ImageOps.equalize(gray_img.resize((32,32)))

class GTRSB_old(Dataset):
    def __init__(self,dir_path,transform=None) -> None:
        super(GTRSB,self).__init__()
        img_paths = [] # images
        labels = [] # corresponding labels
        ROI = [] #region of interest, in the form of(int,int,intint)
        img_size = []
        for c in range(0,43):
            prefix = os.path.join(dir_path,format(c, '05d'))
            
            if os.path.isdir(prefix):
                gtFile_name = 'GT-'+ format(c, '05d') + '.csv'
                gtFile = open(os.path.join(prefix, gtFile_name) )  # annotations file
                gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
                 # skip header
                # loop over all images in current annotations file
                for i,row in enumerate(gtReader):
                    if i !=0:
                        img_paths.append( os.path.join(prefix,row[0])  ) # the 1th column is the filename
                        img_size.append((row[1],row[2])) #(w,h)
                        ROI.append((row[3],row[4],row[5],row[6])) #(x1,y1,x2,y2)
                        labels.append(int(row[7])) # the 8th column is the label
                gtFile.close()
        
        self.img_paths = img_paths
        self.labels = labels
        self.ROW = ROI
        self.img_size = img_size
        self.transform = transform
    def  __len__(self):
        return self.img_paths.__len__()
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        if self.transform is not None:
            rgb_img = Image.open(img_path)
            img =  self.transform(ImageOps.grayscale(rgb_img))
        else:
            raise ValueError("no transforms")
        
        return {'images': img,'labels':self.labels[index] } ###seems that labels accept the form of single integer
    


