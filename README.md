# Autonomy-2024Spring-FP
Hello everyone! This is Jack from GIEE and Tiana from ESOE. In the final project, we are determined to go beyond theory survey and actually make a movable car. Although we didn't make what we expect at the end, it's still our honor to present the work here. 

##    Environment setting
First clone our project to your PC.
```git clone https://github.com/jackmafan/Autonomy-2024Spring-FP.git```

Then change to the main directory, setting the environment by running the following commands. Make sure you have installed anaconda in advance.
```
conda create --name ItA24_FP python=3.9 -y
conda activate ItA24_FP
pip install -r requirements.txt
```
##    Dataset Prepararion
###    GTSRB
The German Traffic Sign Recognition Benchmark (GTSRB) is a widely used dataset in the field of computer vision and machine learning, specifically for traffic sign recognition tasks. It was introduced as part of a competition at the International Joint Conference on Neural Networks (IJCNN) in 2011 to advance the development and benchmarking of traffic sign recognition algorithms.

<div align=center>
  
![GTSRB](assets/figures-in-report/Figure1-1.png)
<div align=left>
  
You can download the dataset through https://benchmark.ini.rub.de/ .
  
After unzip the folder, please put the data under ```GTSRB/``` like the following structures.
```
GTSRB
├───00000
├───00001
...
...
└───00042
```
    
###    CityScapes
The Cityscapes Dataset focuses on semantic understanding of urban street scenes. There are more than 5,000 fine and 20,000 coarse annotated images with fine annotations  more than 50 cities. You can download the dataset through [this link](https://www.cityscapes-dataset.com/) after registration.
<div align=center>
    
![CityScapes](https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/muenster00.png)

<div align=left>
  
Since the images files in folder```CityScapes/``` are just for generating negative samples, you can any negative samples(at least 2048p X 1024p **.png** files) without traffic signs in ```CityScapes/```, too. Our code could traverse the entire folder.

##  Evaluation and Demo
In ```eval.py```, we provide 3 functions: ```testTSR```, ```testLD```, and```evalModel``` to test our implentation. 
    
    
To test ```testTSR``` or ```testLD```, please append arguments ```--testTSR``` or ```--testLD``` after ```python3 eval.py```. Furthermore, remember to specify the paths of tested images end with ```.jpg``` in the list when calling them. The result will be saved in folder ```demo/```
```python=
#...
if args.testTSR:
    testTSR([$PATHS_OF_YOUR_IMAGES])
if args.testLD:
    testLD([$PATHS_OF_YOUR_IMAGES])
#...
       
```

Function ```evalModel``` can test only the classifier itself over **GTSRB** dataset. Argument```--noROI``` uses the entire image rather than image, while ```--testAll``` runs evaluation over entire dataset rather than only validation set. You should name your model weight as ```model_best.pt``` under folder```checkpoints/```.

<div align=center>
  
<img src="assets/road10.jpg" width="250px" height='125px' />
<img src="demo/road10_bin.jpg" width="250px" height='125px' />
<img src="demo/road10_LD.jpg" width="250px" height='125px' />

---

<img src="assets/test0.jpg" width="150px" height='150px' />
<img src="demo/test0_TSR.jpg" width="150px" height='150px' />
<img src="assets/test2.jpg" width="150px" height='150px' />
<img src="demo/test2_TSR.jpg" width="150px" height='150px' />

---
<img src="assets/test1.jpg" width="225px" height='150px' />
<img src="demo/test1_TSR.jpg" width="225px" height='150px' />

<div align=left>

##    Training
Make sure you have prepared datasets well in advance.
###    Detector
First download and install [OpenCV](https://opencv.org/releases/) 3.4. Check whether there are ```opencv_createsamples``` and ```opencv_traincascade``` in ```$YOUR_INSTALL_PATH/opencv/build/x64/vc15/bin```

Please run command ```python detector.py --collect```. You should get the file structure as below.
```
...
├───forbids
│   ├───negative/
│   ├───positive/
│   ├───negative.txt
│   └───positive.txt
├───others
│   ├───negative/
│   ├───positive/
│   ├───negative.txt
│   └───positive.txt
├───suggests
│   ├───negative/
│   ├───positive/
│   ├───negative.txt
│   └───positive.txt
├───warnings
│   ├───negative/
│   ├───positive/
│   ├───negative.txt
│   └───positive.txt
...
```
Copy these folders to ```$YOUR_INSTALL_PATH/opencv/build/x64/vc15/bin```
which looks like
```
...
├───forbids/
├───others/
├───suggests/
├───warnings/
├───opencv_createsamples
├───opencv_traincascade
...
```
Finally, **enter 4 folders one by one** and execute the following commands.
```bash=
 ../opencv_createsamples -info positive.txt -bg negative.txt \
-vec pos.vec \
-num $NUM_OF_POSITIVE_SAMPLES \
-w 32 -h 32
```
```bash=
 ../opencv_traincascade -data ./ -vec pos.vec -bg negative.txt \
-numPos $POS_NUM_EACH_STAGE \
-numNeg $NEG_NUM_EACH_STAGE \
-w 32 -h 32 \
--numStages $YOUR_NUM_OF_STAGES
```
Copy the 4 ```cascade.xml``` files back to folder ```checkpoints/``` and modify their names as in repository. 
###    Classifier
You can directly run command ```python3 train.py``` or slightly adjust the training arguments as you wish.
The result of training will be saved at a unique directory named with ```TSC_$TRAINING_DATETIME``` in folder ```experiment/``` .
    
<div align=center>
    
<img src="assets/figures-in-report/Figure 4-1.png" width="300px" height='225px' />
<img src="assets/figures-in-report/Figure 4-2.png" width="300px" height='225px' />

---
    

<img src="assets/figures-in-report/Figure 4-3.png" width="300px" height='225px' />
<img src="assets/figures-in-report/Figure 4-4.png" width="300px" height='225px' />

<div align=left>
    
## Credits to
###    Collaborator
@[jackmafan](https://github.com/jackmafan)
@[tiana888](https://github.com/tiana888)
###    Reference
####    A Real-Time Traffic Sign Recognition Method Using a New Attention-Based  Deep Convolutional Neural Network for Smart Vehicles [link](https://www.mdpi.com/2076-3417/13/8/4793)
####    Lane detection using color-based segmentation [link](https://scholar.nycu.edu.tw/en/publications/lane-detection-using-color-based-segmentation)
####    German Traffic Sign Recognition Benchmark[link](https://benchmark.ini.rub.de/)
####    The Cityscapes Dataset[link](https://www.cityscapes-dataset.com/)
