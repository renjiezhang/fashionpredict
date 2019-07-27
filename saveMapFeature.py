import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile


# Inception-v3 initialization
BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
IMAGE_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
MODEL_DIR = 'model/'
MODEL_FILE = 'tensorflow_inception_graph.pb'

# image folder
INPUT_DATA = 'fashion_data/walmartTshirt/'
FEATURE_DIR = 'feature_maps/'
 
file_list=[]
for x in os.walk(INPUT_DATA):
    file_list=x[2]
        
       
# get image files
def create_image_lists(sess,bottleneck_tensor,image_data_tensor):

    for file_name in file_list:
        
        file_feature_name=str.replace(file_name,'.jpg','.txt')
        file_feature_name=str.replace(file_feature_name,'.JPG','.txt')  
        file_feature_name=str.replace(file_feature_name,'.jpeg','.txt') 
        if not os.path.exists(FEATURE_DIR+file_feature_name):  
            
            image_data = gfile.FastGFile(INPUT_DATA+file_name, 'rb').read()
            bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
            bottleneck_values = np.squeeze(bottleneck_values)   

            bottleneck_string = ','.join(str(x) for x in bottleneck_values)
            with open(FEATURE_DIR+file_feature_name, 'w') as bottleneck_file:
                bottleneck_file.write(bottleneck_string)           
          
	#read inception v3 data
with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
# get inception bottleneck
bottleneck_tensor, image_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, IMAGE_DATA_TENSOR_NAME])


with tf.Session() as sess:
    tf.global_variables_initializer().run()
     # read all images
    create_image_lists(sess,bottleneck_tensor,image_data_tensor)

