import streamlit as st 

from os import path 
from glob import glob 
from PIL import Image 

with st.sidebar:
    st.header("uploaded image")

i_path = None 
root = 'images_store'
btn = None 


btn = st.button('refresh')
if btn is not None:
    image_paths = glob(path.join(root, '*.jpg'))
    image_paths.insert(0, '...')
    i_path = st.selectbox('generated images', image_paths)
    if i_path is not None: 
        if path.isfile(i_path):
            content = Image.open(i_path).convert('RGB')
            st.image(content)
        print(i_path)

    btn = None 