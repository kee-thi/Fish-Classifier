import streamlit as st
import numpy as np
from numpy import asarray
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import tensorflow as tf
# st.set_page_config(page_title='C-13 Mini project', page_icon='😁')
# from keras.preprocessing.image import img_to_array
# from tensorflow.keras.utils import img_to_array
model = load_model("model2.h5")
def predict(file_path):
    image = Image.open(file_path)
    # x=np.array(image)
    # x = x.resize((224,224))
    # x = img_to_array(imge)
    x = asarray(image)
    x = x / 255.
    # x = cv2.resize(x,(224, 224))
    x = cv2.resize(x,(224, 224), 3)
    x = np.expand_dims(x, axis=0)

    predi = model.predict(x)
    print(predi)
    classes_x = np.argmax(predi)
    print(classes_x)

    classes = ["Black Sea Sprat", "Gilt-Head Bream", "Hourse Mackerel", "Red Mullet", "Red Sea Bream ", "Sea Bass ",
               "Shrimp", "Striped Red Mullet", "Trout"]
    prediction_label = classes[classes_x]
    return prediction_label

st.title('Fish Classification')

 
# st.markdown(""" <style>

# MainMenu {visibility: hidden;}
# header {visibility:hidden;}
# footer {visibility: hidden;}
# [data-testid="stAppViewContainer"]{
# background-image: url("https://img.freepik.com/free-vector/school-fishes-background-hand-drawn-style_23-2147792684.jpg?w=740&t=st=1678698258~exp=1678698858~hmac=b416713dddab1e9756ff67c5fea91ecf635f15c4b7af2805d070d2f1a5f3b4a6");
# background-color:cover;
# background-image : position-absolute;
# background-image: display-flex;
# }

# </style> """, unsafe_allow_html=True)

imge=st.file_uploader('Upload your file', type=['JPG', 'PNG', 'JPEG', 'TIFF'], accept_multiple_files=False, key=None, help=None,
                 on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")


if(imge!=None):
    st.image(imge, caption='Uploaded Image')

if st.button('Predict'):

    predict=predict(imge)
    st.markdown(""" <style> .predict {
font-size:50px ; font-family: 'Cooper Black'; color: #FF9633;} 
</style> """, unsafe_allow_html=True)
    st.write(predict)
