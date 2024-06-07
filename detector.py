import argparse
import csv
import os
import random
from PIL import Image
from tqdm import tqdm
import cv2

def collect():
    """collect the positive and negative samples to correct directories"""
    random.seed(0) ##fix random seed
    
    #### GENERATE POSITIVE IMAGES ####
    forbids = [0,1,2,3,4,5,7,8,9,10,15,16] + [6,17,32,41,42]
    warnings = [11,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    suggests = [33,34,35,36,37,38,39,40]
    others = [12,13,14]
    
    folders = ['forbids','warnings','suggests','others']
    groups = [forbids,warnings,suggests,others]
    pos_imgs_count = [0,0,0,0]
    
    for groupid,group in enumerate(groups):       
        save_prefix = os.path.join(folders[groupid],'positive')
        os.makedirs(save_prefix,exist_ok=True)
        
        pos_f = open(os.path.join(folders[groupid],'positive.txt'),'w')
        
        id = 0
        for c in tqdm(group):
            ## get gt file
            prefix = os.path.join('GTSRB',format(c, '05d'))
            gtFile_name = 'GT-'+ format(c, '05d') + '.csv'
            gtFile = open(os.path.join(prefix, gtFile_name) )  # annotations file
            gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
                        
            # skip header
            # loop over all images in current annotations file
            for _,row in enumerate(gtReader):
                if _ !=0:
                    if random.random() < 0.2 : #keep only 1/5 images for training
                        img_path = os.path.join(prefix,row[0])
                        img = Image.open(img_path)
                        x1,y1,x2,y2 = int(row[3]),int(row[4]),int(row[5]),int(row[6])
                        x,y,w,h = x1,y1,x2-x1+1,y2-y1+1
                        
                        img.save(os.path.join(save_prefix,f'{id}.png'))
                        pos_f.write(os.path.join('.','positive',f'{id}.png')+f' 1 {x} {y} {w} {h}\n')
                        img.close()
                        pos_imgs_count[groupid] +=1
                        id = id+1
            gtFile.close()
    
        pos_f.close()
    #### GENERATE NEGATIVE IMAGES ####

    # collect negative file paths
    jpg_files = []
    for root, _, files in os.walk('CityScapes'):
        for file in files:
            if file.lower().endswith(('.png')):
                jpg_files.append(os.path.join(root,file))    

    crop_option = [(96,96),(192,192),(256,256)]
    for groupid,folder in enumerate(folders):
        save_prefix = os.path.join(folder,'negative')
        os.makedirs(save_prefix,exist_ok=True)
        
        neg_f = open(os.path.join(folders[groupid],'negative.txt'),'w')
        
        for id in tqdm(range(2*pos_imgs_count[groupid])):# for each cascaded classifier prepare 4 times of negative label
            # get random sample parameters
            img_path = random.choice(jpg_files)
            w,h = random.choice(crop_option)
            x,y = random.randint(500,1500),random.randint(250,750)
            x1,y1,x2,y2 = x,y,x+w-1,y+h-1
            
            #get random negative sample
            img = Image.open(img_path)
            img = img.crop((x1,y1,x2,y2))
            #img = img.resize((32,32))
            img.save(os.path.join(save_prefix,f'{id}.png'))
            neg_f.write(os.path.join('.','negative',f'{id}.png')+'\n')
            img.close()

        neg_f.close()

def train():
    """train 4 classifiers from the samples in predefined directory"""
    """Umimplementable through python code"""
    """https://blog.csdn.net/u012905422/article/details/77478278"""
    pass
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect',action='store_true',default=False)
    parser.add_argument('--train',action='store_true',default=False)
    args = parser.parse_args()
    
    if args.collect:
        collect()
    if args.train:
        train()
