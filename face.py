import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect(img):
    image = np.array(img.convert('RGB'))
    faces = face_cascade.detectMultiScale(image, 1.05, 5) # scalefactor = 1.05 minNeighbor=5
    for (x,y,w,h) in faces:
        cv2.rectangle(img=image,pt1=(x,y),pt2=(x+w,y+h),color=(255,0,0),thickness=1)
    return image,faces

def main():
    ''' 
    Face Feature detection app 
    '''
    st.title('Sdrawkcab')
    t = "<h2>Face Feature Detection App</h2>"
    st.markdown(t,unsafe_allow_html=True)
    image_file = st.file_uploader("Upload Image",type=['jpeg','png','jpg'])
    

    if image_file is not None:
        image = Image.open(image_file)
        if st.button('Execute'):
            img,res=detect(image)
            st.image(img,use_column_width=True)
            st.success("Found {} faces\n".format(len(res)))
if __name__ == "__main__":
    main()

