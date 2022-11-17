from flask import Flask,render_template,request
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow

app = Flask(__name__)
model=load_model("naturaldisaster.h5")
#print(model)

@app.route('/',methods=['GET'])
def index():
  return render_template('home.html')

@app.route('/home.html',methods=['GET'])
def home():
  return render_template('home.html')

@app.route('/intro.html',methods=['GET'])
def about():
  return render_template('intro.html')

@app.route('/upload.html',methods=['GET'])
def upload():
  return render_template('upload.html')

@app.route('/uploader.html',methods=['GET','POST'])
def predict():
  if request.method == "POST":
    f = request.files['image']
    basepath=os.path.dirname(__file__)
    filepath=os.path.join(basepath,'uploader',f.filename)
    f.save(filepath)
    img=image.load_img(filepath,target_size=(64,64))
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    pred=np.argmax(model.predict(x),axis=1)
    index=['Cyclone','Earthquake','Flood','Wildfire']
    text="The Classified Disaster is:" +str(index[pred[0]])
  return text
if __name__ == '__main__':
  app.run(host='0.0.0.0',port=8000,debug=False)

