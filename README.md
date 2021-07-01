# Instruction
## About **/data** folder
- First we store the data for training in the **/data/train** folder. 
- One folder of images created in this folder will stand for one class.
- You can follow the format of the **/data** folder in this repositery.
## Traning step 
- Installing dependency:
```
pip install --upgrade pip
pip install tensorflow
pip install scikit-image
pip install numpy
pip install opencv-python
```
- Traing
```
python train.py
```
- Remember to change the variable **num_class_output** in the file **train.py** to your number of classes.
- After training, the model will be stored in the folder **/Trained_model**, the indices of classes will be store at **class_indices.json** file 
## Using your model
- Change the variable **url** in the **test.py** file, and then run this file to get the predicted class of the url image.
```
python test.py
```
