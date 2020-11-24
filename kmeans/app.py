import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import cv2 as cv

@st.cache
def cal_dis_model(img,max_k):
    img = img.copy()
    s = img.shape
    w = 500 
    h = int(w*s[0] / s[1])
    img = cv.resize(img,(w,h))
    img = img.reshape(-1,3)

    dis = []
    models = []
    for k in range(1,max_k+1):
        model = KMeans(k, max_iter=10)
        model.fit(img)
        dis.append(model.inertia_)
        models.append(model)
        placeholder_1.text("Running: "+str(k)+"/"+str(max_k))
        placeholder_2.progress(k/max_k)
    return dis,models,(w,h)
    
@st.cache
def process_img(models,i,shape):
    model = models[i]
    processed_img = np.zeros((shape[0]*shape[1],3))
    for i in range(model.cluster_centers_.shape[0]):
        processed_img[model.labels_ == i,:] = model.cluster_centers_[i]
    processed_img = processed_img.astype("int")
    processed_img = processed_img.reshape((shape[1],shape[0],3))
    return processed_img

@st.cache
def reduce_img(img):
    s = img.shape
    w = 698 
    h = int(w*s[0] / s[1])
    reduced_img = cv.resize(img,(w,h))
    return reduced_img

# Input image
uploaded_file = st.file_uploader("Image:", type=["png", "jpg"])

# Input max K
max_k = st.number_input("Select max k:",0,200,10)

# Selected k 


models = None

placeholder_1 = st.empty()
placeholder_2 = st.empty()

# Display image
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img = np.array(img)
    st.image(reduce_img(img), width=698)
    placeholder_1.text("Running...")
    placeholder_2.progress(0)
    dis,models,shape = cal_dis_model(img,max_k)
    st.line_chart(dis)

selected_k = st.slider("Choose k: ", 1, max_k, 5)

if models is not None:
    processed_img = process_img(models,selected_k-1,shape)
    st.image(processed_img, width=698)
