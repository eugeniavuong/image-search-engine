import csv
from glob import glob
from pathlib import Path
from statistics import mean

from towhee import pipe, ops, DataCollection
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# Towhee parameters
MODEL = 'resnet50'
DEVICE = None # if None, use default device (cuda is enabled if available)

# Milvus parameters
HOST = '127.0.0.1'
PORT = '19530'
TOPK = 10
DIM = 2048 # dimension of embedding extracted by MODEL
COLLECTION_NAME = 'reverse_image_search'
INDEX_TYPE = 'IVF_FLAT'
METRIC_TYPE = 'L2'

# path to csv (column_1 indicates image path) OR a pattern of image paths
INSERT_SRC = 'reverse_image_search.csv'
QUERY_SRC = './test/*/*.JPEG'

# Create a milvus collection inside the DB
def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name) #if the collection exists we will delete it first 
    
    fields = [
        FieldSchema(name="path", dtype=DataType.VARCHAR, description="path to image", max_length=500,
                is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, description="image embedding vectors", dim= dim)
    ]
    schema = CollectionSchema(fields, description="reverse image search")
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        'metric_type': METRIC_TYPE,
        'index_type': INDEX_TYPE,
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name='embedding', index_params=index_params)
    return collection

# Connect to Milvus with host and portt and create a collection - open this in Attu (GUI to interact with the db)
connections.connect(host=HOST, port=PORT)

# Create collection 
collection = create_milvus_collection(COLLECTION_NAME, DIM)
print(f'A new collection created: {COLLECTION_NAME}')


"""Create embedding pipeline passing the image path into the image embedding operator"""

# Create vectors for the images 

#load image path
def load_image(file):
    if file.endswith('csv'):
        with open(file) as f:
            reader = csv.reader(f)
            next(reader)
            for item in reader:
                yield item[1]
    else:
        for item in glob(file):
            yield item

# Embedding pipeline
p_embed = (
    pipe.input('src') #image file path(s) from reversie_image_search.csv
        .flat_map('src', 'img_path', load_image) # load image data
        .map('img_path', 'img', ops.image_decode()) # decodes image into readable format for the CNN
        .map('img', 'vec', ops.image_embedding.timm(model_name=MODEL, device=DEVICE)) # extract the embeddings 
)

# Display embedding result, no need for implementation
p_display = p_embed.output('img_path', 'img', 'vec')
DataCollection(p_display('./test/goldfish/*.JPEG')).show()