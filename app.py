import csv
import streamlit as st
from glob import glob
from pathlib import Path
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility



# Milvus parameters
HOST = '127.0.0.1'
PORT = '19530'
DIM = 2048 # dimension of embedding extracted by MODEL
COLLECTION_NAME = 'reverse_image_search'
INDEX_TYPE = 'IVF_FLAT'
METRIC_TYPE = 'L2'

# Path to csv or image pattern
INSERT_SRC = 'reverse_image_search.csv'

# Connect to Milvus
connections.connect(host=HOST, port=PORT)

# Function to create Milvus collection
def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
    else:
    
        fields = [
            FieldSchema(name="path", dtype=DataType.VARCHAR, description="path to image", max_length=500, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, description="image embedding vectors", dim=dim)
        ]
        schema = CollectionSchema(fields, description="reverse image search")
        collection = Collection(name=collection_name, schema=schema)

        index_params = {'metric_type': METRIC_TYPE, 'index_type': INDEX_TYPE, 'params': {"nlist": 2048}}
        collection.create_index(field_name='embedding', index_params=index_params)

    return collection

# Load images from file or csv
def load_image_paths(file):
    if file.endswith('csv'):
        with open(file) as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                yield row[1]
    else:
        for file in glob(file):
            yield file

# Function to extract embeddings
def extract_embedding(image_path):
    # PyTorch model for embedding extraction
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # remove last (classification layer)
    model.eval()

    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # normalise images using the mean and st of RGB colour channels
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().numpy()  # Remove unnecessary dimensions
    return embedding

# Function to manually insert into Milvus
def insert_into_milvus(collection, image_paths):
    paths = []
    embeddings = []

    for image_path in image_paths:
        embedding = extract_embedding(image_path)
        paths.append(image_path)
        embeddings.append(embedding)

    # Insert data into the Milvus collection
    collection.insert([paths, embeddings])
    collection.flush()  # Ensure data is written to the DB

    print(f'Inserted {len(paths)} images into the collection.')


def search_similar_images(collection, image_path, top_k=5):
    # Extract the embedding of the query image 
    print(f"Exctracting embedding of query: {image_path}")
    query_embedding = extract_embedding(image_path)

    print(f"Query embedding: {query_embedding[:5]}...")  # Print first 5 elements of the embedding for inspection


    if query_embedding is None or len(query_embedding) ==0:
        print("Error: No embedding was extracted")
        return[]

    # Search in Milvus for the top_k 
    try:
        search_params = {"metric_type" : METRIC_TYPE, "params":{"nprobe": 100}} # L2 - euclidean distance, nprobe is number of units to query during search
        search_result = collection.search(
            data=[query_embedding], # Input embedding to search
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            out_fields=["path"] # retrieve the 'path' field
        )
    
        print("Raw search results:", search_result)  # Print the raw search results for inspection
    except Exception as e:
        print(f"Error during search: {e}")
        return[]
    
    print(f"Number of similar images found: {len(search_result[0])}")

    # Process and print the top-k similar images
    similar_image_paths =[]
    print(f"Top {top_k} similar images:")
    for result in search_result:
        if len(result) > 0:
            for match in result:
                similar_image_paths.append(match.id)
                print(f"Image path: {match.id}, Distance: {match.distance}")
        else:
            print("No similar images found.")
    return similar_image_paths
"""
    # Retrieve the path of the top simialar images
    similar_image_paths =[]
    for result in search_result:
        similar_image_paths.append(result.entity.get("path"))
    return similar_image_paths
"""

# Function to load the collection 
def load_collection(collection):
    if not utility.has_collection(collection.name):
        raise Exception(f"Collection {collection.name} does not exist.")
    
    # Load the collection into memory 
    if not collection.is_loaded:
        collection.load()
    print(f"Collection '{collection.name}' loaded into memory")

def mainF(image_path):
    # Create the Milvus collection
    collection = create_milvus_collection(COLLECTION_NAME, DIM)

    # Load image paths and manually insert into Milvus
    image_paths = list(load_image_paths(INSERT_SRC))
    insert_into_milvus(collection, image_paths)

    # Check number of entities in the collection
    print('Number of entities in the collection:', collection.num_entities)

    collection = Collection(COLLECTION_NAME)

    load_collection(collection)

    #image_path = './test/goldfish/n01443537_3883.JPEG'

    # Get the top 5 similar images
    similar_images = search_similar_images(collection, image_path, top_k=5)

    print("Top 5 similar images:")
    for img_path in similar_images:
        print(img_path)

    return similar_images # returns the pathers of the top

