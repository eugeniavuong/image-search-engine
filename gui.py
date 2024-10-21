import streamlit as st

st.title("Image Search Engine :female-detective:")
st.markdown("This image search engine allows you to upload an image to find similar images in the database.")

uploaded_files = st.file_uploader(
    "Choose a file", accept_multiple_files=True
)