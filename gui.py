import streamlit as st
import os, tempfile
from app import *

st.title("Image Search Engine :female-detective:")
st.markdown("This image search engine allows you to upload an image to find similar images in the database.")

uploaded_file = st.file_uploader(
    "Choose a file", type="JPEG"
)
submit=st.button("Search")

if submit:
    temp_dir = tempfile.mkdtemp() # create temp file directory
    path = os.path.join(temp_dir, uploaded_file.name) # concatenate the temp file directory with filename
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())

    top_images = mainF(path)
    st.header("Top 5 similar images:")

    if top_images:
        for image in top_images:
            st.image(image, use_column_width=True)
    else:
        st.write("No similar images found.")
