import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import csv


INCEPTION_FEATURE_SIZE = 2048
INCEPTION_POOL_NAME = 'pool_3/_reshape:0'
IMAGE_DATA_NAME = 'DecodeJpeg/contents:0'
MODEL_DIR = 'model/'
MODEL_FILE = 'tensorflow_Inception_graph.pb'

# image folder
INPUT_DATA = 'fashion_data/walmartTshirt/'
FEATURE_DIR = 'feature_maps/'
logs_dir='./event_logs'

POPULARREVIEW=50
TEST_PERCENTAGE = 30

#network setting
LEARNING_RATE = 0.001
STEPS = 4000
BATCH = 10
PRICE_FILTER=0

x_training=[]
y_training=[]
x_validation=[]
y_validation=[]
x_testing=[]
y_testing=[]
y_training_price=[]
x_testing_price=[]
y_training_rate=[]
x_testing_rate=[]
y_training_extra_features=[]
x_testing_extra_features=[]

label_list=[]


with open('fashion_data/walmartTshirt.csv', 'r') as f:
  reader = csv.reader(f)
  label_list = list(reader)

  
file_list=[]
for x in os.walk(INPUT_DATA):
    file_list=x[2]
        
def get_review(filename):
    for l in label_list:
        if(l[0].strip()==filename.strip()):
            return l[1]
        
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
        
# get image files
def create_image_lists(sess,Inception_tensor,image_data_tensor):

    for file_name in file_list:
        
            file_feature_name=str.replace(file_name,'.jpg','.txt')
            file_feature_name=str.replace(file_feature_name,'.JPG','.txt')  
            file_feature_name=str.replace(file_feature_name,'.jpeg','.txt')
            fileID=str.replace(file_name,'.jpg','')
            fileID=str.replace(fileID,'.JPG','')            
            fileID=''.join(x for x in fileID if x.isdigit())  
            
            if os.path.exists(FEATURE_DIR+file_feature_name):        
    
                reviews=get_review(fileID)
                price=get_price(fileID)
                rate=get_rate(fileID)
                sleeve=get_sleeve(fileID)
                netcut=get_netcut(fileID)
                length=get_length(fileID)
                back=get_back(fileID)
        

                
                if reviews is None:
                    reviews=0
                if rate is None :
                    rate=0
                if price is None:
                    price=0        
                
                #print(price)
                if not price: 
                    continue
                if float(price)<PRICE_FILTER:
                    continue
                
                with open(FEATURE_DIR+file_feature_name, 'r') as Inception_file:
                    Inception_string = Inception_file.read()
                    
                Inception_values = [float(x) for x in Inception_string.split(',')]
                Inception_values = np.squeeze(Inception_values)  
                Inception_values = np.asarray(Inception_values)
                Inception_values=Inception_values.reshape(-1)
                
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

                x_value=np.concatenate((sleeve,netcut,length,back), axis=0)
                x_value=x_value.reshape(-1) 

                chance = np.random.randint(100) 
                if chance > TEST_PERCENTAGE:
                    x_training.append(Inception_values)
                    y_training.append(reviews) 
                    y_training_price.append(price)
                    y_training_rate.append(rate)
                    y_training_extra_features.append(x_value)
        
                else:
                    x_testing.append(Inception_values)
                    y_testing.append(reviews)        
                    x_testing_price.append(price)
                    x_testing_rate.append(rate)
                    x_testing_extra_features.append(x_value)
            
def get_random_training_data():
    selectedX = []
    SelectedY = []
    SelectedPrice = []
    SelectedRate=[]
    SelectedExtra=[]
    
    data_length=len(x_training)

    for _ in range(BATCH):
        image_index = random.randrange(data_length)
        
        selectedX.append(x_training[image_index])
        SelectedY.append(y_training[image_index])
        SelectedPrice.append(y_training_price[image_index])
        SelectedRate.append(y_training_rate[image_index])
        SelectedExtra.append(y_training_extra_features[image_index])
    return selectedX, SelectedY,SelectedPrice,SelectedRate,SelectedExtra

def get_random_validation_data():
    selectedX = []
    SelectedY = []
    data_length=len(x_validation)
    for _ in range(BATCH):
        image_index = random.randrange(data_length)

        selectedX.append(x_validation[image_index])
        SelectedY.append(y_validation[image_index])
    return selectedX, SelectedY

def get_accurcy(predict,groundtruth):
    dataLenth=len(predict)
    predict_pop=groundtruth_pop=0
    equal_count=0
    for i in range(0,dataLenth-1):

        if int(predict[i])>=100:
            predict_pop=2
        elif int(predict[i])>=50:
            predict_pop=1
        else:
            predict_pop=0
            
        if int(groundtruth[i])>=100:
            groundtruth_pop=2
        elif int(groundtruth[i])>=50:
            groundtruth_pop=1
        else:
            groundtruth_pop=0
        
        if predict_pop==groundtruth_pop:
            equal_count+=1
    accuracy=equal_count/dataLenth
    return accuracy
        

def get_mape(y_pred,y_groundtruth): 

    mape=np.mean(np.abs((y_groundtruth - y_pred) / y_groundtruth + 1e-10))  
    mape=np.round(mape,decimals=4)
    return   mape

def get_mse(y_pred,y_groundtruth): 

    mse = (np.square(y_groundtruth - y_pred)).mean()
   
    return   mse
    
	#read Inception v3 data
with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  

Inception_tensor, image_data_tensor = tf.import_graph_def(graph_def, return_elements=[INCEPTION_POOL_NAME, IMAGE_DATA_NAME])

X_input = tf.placeholder(tf.float32, [None, INCEPTION_FEATURE_SIZE], name='Inception')
# ground truth
Y_input = tf.placeholder(tf.float32, [None, 1], name='GroundTruth')

X_price = tf.placeholder(tf.float32, [None, 1], name='Price')

X_rate = tf.placeholder(tf.float32, [None, 1], name='Rate')

X_extra=tf.placeholder(tf.float32, [None, 4], name='Extra')

weights1 = tf.Variable(tf.truncated_normal([INCEPTION_FEATURE_SIZE, 10], stddev=0.001))
weights2 = tf.Variable(tf.truncated_normal([1, 10], stddev=0.01))
weights3 = tf.Variable(tf.truncated_normal([4, 10], stddev=0.1))
weights4 = tf.Variable(tf.truncated_normal([1, 10], stddev=0.01))
biases = tf.Variable(tf.zeros([10]))

#regression = tf.matmul(X_input, weights1) +tf.matmul(X_price, weights2)+tf.matmul(X_extra, weights3)+tf.matmul(X_rate, weights4) + biases
regression = tf.matmul(X_input, weights1) +tf.matmul(X_price, weights2)+tf.matmul(X_extra, weights3) + biases

final_tensor = tf.layers.dense(regression, 10, tf.nn.relu)         
output = tf.layers.dense(final_tensor, 1)                    

 
loss = tf.losses.mean_squared_error(Y_input, output)
#loss = tf.reduce_mean(tf.abs(tf.divide(tf.subtract(output,Y_input),(Y_input + 1e-10))))


optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss)

trans_Y_input = tf.placeholder(tf.float32, [None, 1], name='trans_Y_input')
trans_output = tf.placeholder(tf.float32, [None, 1], name='trans_output')

with tf.name_scope('evaluation'):
    correct_prediction = tf.equal(tf.argmax(trans_output, 1), tf.argmax(trans_Y_input, 1))
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
     # read all images
    create_image_lists(sess,Inception_tensor,image_data_tensor)

    y_training=np.asanyarray(y_training)
    y_training=y_training.reshape(len(y_training),1)
    
    y_training_price=np.asanyarray(y_training_price)
    y_training_price=y_training_price.reshape(len(y_training_price),1)

    y_training_rate=np.asanyarray(y_training_rate)
    y_training_rate=y_training_rate.reshape(len(y_training_rate),1)
    
    y_testing=np.asanyarray(y_testing)
    y_testing=y_testing.reshape(len(y_testing),1)

    y_training_extra_features=np.asanyarray(y_training_extra_features)
    y_training_extra_features=y_training_extra_features.reshape(len(y_training_extra_features),4)
    
    x_testing_price=np.asanyarray(x_testing_price)
    x_testing_price=x_testing_price.reshape(len(x_testing_price),1)
    
    x_testing_rate=np.asanyarray(x_testing_rate)
    x_testing_rate=x_testing_rate.reshape(len(x_testing_rate),1)
    
    x_testing_extra_features=np.asanyarray(x_testing_extra_features)
    x_testing_extra_features=x_testing_extra_features.reshape(len(x_testing_extra_features),4)
    
    for i in range(STEPS):
        selectedX, selectedY,selectedPrice,SelectedRate,SelectedExtra = get_random_training_data()
        _, train_loss, pred = sess.run([train_op, loss, output],feed_dict={X_input: selectedX, Y_input: selectedY, X_price:selectedPrice,X_rate:SelectedRate,X_extra:SelectedExtra})
        
        trans_pred=np.asarray(pred,dtype=np.float32).reshape(-1)
        trans_selectedY=np.asarray(selectedY,dtype=np.float32).reshape(-1)
        accuracy=get_accurcy(trans_pred,trans_selectedY)
        mape=get_mape(trans_pred,trans_selectedY)
        #mse=get_mse(trans_pred,trans_selectedY)
#        evaluation_step = sess.run([evaluation_step],feed_dict={trans_output: trans_output, trans_Y_input: trans_selectedY})        
        if i%10 == 0 or i+1 == STEPS:
              #print('Step ',i, 'MAPE: ', train_loss,'MSE: ',mse)
              print('Step ',i, 'MSE: ', train_loss,'MAPE', str(mape))



    # final accuracy
    print('')
    print('________________Testing _______________')

    _, testing_loss, testing_pred = sess.run([train_op, loss, output],feed_dict={X_input: x_testing, Y_input: y_testing,X_price:x_testing_price, X_rate:x_testing_rate,X_extra:x_testing_extra_features})
    testing_trans_pred=np.asarray(testing_pred,dtype=np.float32).reshape(-1)
    testing_trans_selectedY=np.asarray(y_testing,dtype=np.float32).reshape(-1)
    testing_accuracy=get_accurcy(testing_trans_pred,testing_trans_selectedY)
    #testing_mse=get_mse(testing_trans_pred,testing_trans_selectedY)    
    testing_mape=get_mape(testing_trans_pred,testing_trans_selectedY)  
    print('Testing MAPE: ', testing_mape)
    print('Testing MSE: ', testing_loss)
    print('Popularity Level Accuracy : ',np.round(testing_accuracy,decimals=4))
#    writer=tf.summary.FileWriter(logs_dir,sess.graph)
#    writer.close()
#    saver.save(sess, "savedmodel/FashionPopularity")
#    print('Model saved')
