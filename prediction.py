import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image
import matplotlib.pyplot as plt
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
IMAGE_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
MODEL_DIR = 'model/'
MODEL_FILE = 'tensorflow_inception_graph.pb'
N_Classes=1
LEARNING_RATE = 0.001


tf.reset_default_graph()
file_list=[]
fileID='15'
TESTINGPATH='fashion_data/predict/'+(fileID)+'/'

for x in os.walk(TESTINGPATH):
    file_list=x[2]
def getPredictImage(sess,bottleneck_tensor,image_data_tensor):
 

    x_predict=[]
    x_price=[]
    x_rate=[]
    x_extra=[]
    price=get_price(fileID)
    rate=get_rate(fileID)
    sleeve=get_sleeve(fileID)
    netcut=get_netcut(fileID)
    length=get_length(fileID)
    back=get_back(fileID)
    price=np.asarray(price)
    price=price.reshape(-1)
    rate=np.asarray(rate)
    rate=rate.reshape(-1)
    sleeve=np.asarray(sleeve)
    sleeve=sleeve.reshape(-1)
    netcut=np.asarray(netcut)
    netcut=netcut.reshape(-1)
    length=np.asarray(length)
    length=length.reshape(-1)
    back=np.asarray(back)
    back=back.reshape(-1)
    extra=np.concatenate((sleeve,netcut,length,back), axis=0)
    extra=extra.reshape(-1)
    for f in file_list :   
        image_data = gfile.FastGFile(TESTINGPATH+f, 'rb').read()
        bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
        bottleneck_values = np.squeeze(bottleneck_values) 
        bottleneck_values = np.asarray(bottleneck_values)
        bottleneck_values=bottleneck_values.reshape(-1)
        x_predict.append(bottleneck_values)
        x_price.append(price)
        x_rate.append(rate)
        x_extra.append(extra)
    return x_predict,x_price,x_rate,x_extra

label_list=[]
with open('fashion_data/walmartPredict.csv', 'r') as f:
  reader = csv.reader(f)
  label_list = list(reader)
  
        
def get_rate(filename):
    for l in label_list:
        if(l[0].strip()==filename.strip()):
            return l[2]
        
def get_price(filename):
    for l in label_list:
        if(l[0].strip()==filename.strip()):
            return l[3]

def get_sleeve(filename):
    for l in label_list:
        if(l[0].strip()==filename.strip()):
            return l[4]

def get_netcut(filename):
    for l in label_list:
        if(l[0].strip()==filename.strip()):
            return l[5]

def get_length(filename):
    for l in label_list:
        if(l[0].strip()==filename.strip()):
            return l[6]

def get_back(filename):
    for l in label_list:
        if(l[0].strip()==filename.strip()):
            return l[7]
        
def get_popularityLevel(predict):

    popularity_level=''
    if int(predict)>=100:
        popularity_level='Very Popular'
    elif int(predict)>=50:
        popularity_level='Popular'
    else:
        popularity_level='Normal'
            
    return popularity_level
        
with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
# get inception bottleneck
bottleneck_tensor, image_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, IMAGE_DATA_TENSOR_NAME])

X_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')

Y_input = tf.placeholder(tf.float32, [None, 1], name='GroundTruth')

X_price = tf.placeholder(tf.float32, [None, 1], name='Price')

X_rate = tf.placeholder(tf.float32, [None, 1], name='Rate')

X_extra=tf.placeholder(tf.float32, [None, 4], name='Extra')

weights1 = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, 10], stddev=0.001))
weights2 = tf.Variable(tf.truncated_normal([1, 10], stddev=0.01))
weights3 = tf.Variable(tf.truncated_normal([4, 10], stddev=0.1))
weights4 = tf.Variable(tf.truncated_normal([1, 10], stddev=0.5))
biases = tf.Variable(tf.zeros([10]))

regression = tf.matmul(X_input, weights1) +tf.matmul(X_price, weights2)+tf.matmul(X_extra, weights3)+tf.matmul(X_rate, weights4) + biases
#regression = tf.matmul(X_input, weights1) +tf.matmul(X_price, weights2)+tf.matmul(X_extra, weights3) + biases

final_tensor = tf.layers.dense(regression, 10, tf.nn.relu)         
output = tf.layers.dense(final_tensor, 1)    

saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
     # read all images
    saver = tf.train.import_meta_graph('savedmodel/FashionPopularity.meta',clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint('savedmodel/'))
    x_predict,x_price,x_rate,x_extra=getPredictImage(sess,bottleneck_tensor,image_data_tensor)
    x_predict=np.asarray(x_predict)


    prediction =  sess.run(output,feed_dict={X_input: x_predict,X_price:x_price, X_rate:x_rate,X_extra:x_extra})
    print(prediction)
    prediction=np.mean(prediction)
    prediction=np.round(prediction,decimals=0)
    
    if prediction==0:
        prediction=1
    popularity_level=get_popularityLevel(prediction)
    fig=plt.figure(figsize=(8, 8))

    for i in range(0, len(file_list)):
        img=mpimg.imread(TESTINGPATH+file_list[i])
        fig.add_subplot(1, 4, i+1)
        plt.axis('off')
        plt.imshow(img)
        plt.show()
    
    print('The estimated number of review is :' )
    print (prediction)
    print('The estimated level of popularity is :' )
    print(popularity_level)


#    for f in range(len(file_list)):
#        print('---------------------')
#
#        salesNumber=get_sales(str.replace(file_list[f],'.jpg',''))
#               
#        print(file_list[f],' : ',prediction[f] , 'sold out sizes :',salesNumber )