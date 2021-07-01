import keras
from keras.models import Model
my_model=keras.models.load_model("Trained_model")

from skimage import io
import cv2
import numpy as np
url="https://i.pinimg.com/originals/4e/2b/c1/4e2bc1412d2f3702e3159aa94a0ca913.jpg"
img=io.imread(url)
img=cv2.resize(img,(240,240))
img=img/255
img=np.array([img])
y_predict=my_model.predict(img)


print('predicted class: '+str(y_predict[0].argmax()))