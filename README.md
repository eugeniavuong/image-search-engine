# image-search-engine

Reverse image search takes an image as input and retrieves most similar images based on its content. The basic idea behind semantic image search is to represent each image as an embedding of features extracted by a pretrained deep learning model. Then image retrieval can be performed by storing & comparing image embeddings.

<img width="937" alt="image" src="https://github.com/user-attachments/assets/b07d664d-5d76-4d9a-8322-d1e5356e3320">


## Preparation 
To set up the project to perform reverse image search you'll need to download some packages and start5 Milvus 

download the packages from the requirements.txt file:

<code>pip install requirements.txt</code>


## Prepare data
For this project we are using a subset from the ImageNet database, the strucutre of the example dataset is as follows:
train: directory of candidate images, 10 images per class from ImageNet train data
test: directory of query images, 1 image per class from ImageNet test data
reverse_image_search.csv: a csv file containing id, path, and label for each candidate image

<code>! curl -L https://github.com/towhee-io/examples/releases/download/data/reverse_image_search.zip -O
! unzip -q -o reverse_image_search.zip </code>

## Start Milvus
This project uses milvus 2.2.10 and pymilvus 2.2.11.

<code>! wget https://github.com/milvus-io/milvus/releases/download/v2.2.10/milvus-standalone-docker-compose.yml -O docker-compose.yml
! docker-compose up -d
! python -m pip install -q pymilvus==2.2.11</code>

### How to run the project 

<code>streamlit run gui.py</code>

### Performance Improvements 
- To improve search performance of the collection, I should work on optimising the parameters affecting the way i'm loading the collaction (e.g. replica, index type)

<img width="557" alt="image" src="https://github.com/user-attachments/assets/43fa7576-ac53-450a-aacd-9a0568d771c9">






