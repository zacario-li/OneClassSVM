# Simple One Class SVM
## requirement
python >=3.6    
keras  
tensorflow >=2.0  
scipy  
sklearn  
opencv  
numpy  
tqdm
## brief
I use tensorflow as backend to implement an image one class svm classifier  
I use ResNet50 as the feature extractor
## How to use
put your images into 'data/train' for training  
put your test images into 'data/test'  
### modify 'ocsvm.py'
I defined two methods for you to easily use the one class svm  
**trainSVM()**  # this is the training function, after finished, 3 model files will be saved.  
**loadSVMAndPredict()**  # this is the predict function.