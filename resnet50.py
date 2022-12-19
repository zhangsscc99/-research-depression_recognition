# %%
# Loading all necessary libraries and modules
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time
# Set the TensorFlow backend to use the GPU
tf.config.experimental.set_visible_devices([], 'GPU')
from matplotlib import gridspec

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
#from keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.callbacks import ModelCheckpoint

#from sklearn.utils import shuffle
#from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow.keras as keras

# %% [markdown]
# # face alignment
print(tf.config.list_physical_devices('GPU'))
# %%
"""

import face_alignment
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

input = io.imread('/Users/mac/Desktop/implementation_depression/20-50/train/20/122542.jpg')
preds = fa.get_landmarks(input)"""

# %% [markdown]
# # model from scratch

# %%
def resnet50_model():
    # Load a model if we have saved one
    #if(os.path.isfile('C:\\DATA\\Python-data\\CIFAR-10\\models\\resnet_50.h5') == True):
    #    return keras.models.load_model('C:\\DATA\\Python-data\\CIFAR-10\\models\\resnet_50.h5')
    # Create an input layer 
    input = keras.layers.Input(shape=(None, None, 3))
    # Create output layers
    output = keras.layers.ZeroPadding2D(padding=3, name='padding_conv1')(input)
    output = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name='conv1')(output)
    output = keras.layers.BatchNormalization(axis=3, epsilon=1e-5, name='bn_conv1')(output)
    output = keras.layers.Activation('relu', name='conv1_relu')(output)
    output = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(output)
    output = conv_block(output, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    output = identity_block(output, 3, [64, 64, 256], stage=2, block='b')
    output = identity_block(output, 3, [64, 64, 256], stage=2, block='c')
    output = conv_block(output, 3, [128, 128, 512], stage=3, block='a')
    output = identity_block(output, 3, [128, 128, 512], stage=3, block='b')
    output = identity_block(output, 3, [128, 128, 512], stage=3, block='c')
    output = identity_block(output, 3, [128, 128, 512], stage=3, block='d')
    output = conv_block(output, 3, [256, 256, 1024], stage=4, block='a')
    output = identity_block(output, 3, [256, 256, 1024], stage=4, block='b')
    output = identity_block(output, 3, [256, 256, 1024], stage=4, block='c')
    output = identity_block(output, 3, [256, 256, 1024], stage=4, block='d')
    output = identity_block(output, 3, [256, 256, 1024], stage=4, block='e')
    output = identity_block(output, 3, [256, 256, 1024], stage=4, block='f')
    output = conv_block(output, 3, [512, 512, 2048], stage=5, block='a')
    output = identity_block(output, 3, [512, 512, 2048], stage=5, block='b')
    output = identity_block(output, 3, [512, 512, 2048], stage=5, block='c')
    output = keras.layers.GlobalAveragePooling2D(name='pool5')(output)
    output = keras.layers.Dense(2, activation='linear')(output)
    # Create a model from input layer and output layers
    model = keras.models.Model(inputs=input, outputs=output)
    # Print model
    print()
    #print(model.summary(), '\n')
    # Compile the model
    model.compile(optimizer='adam',loss='mse')# Return a model
    return model
# Create an identity block
def identity_block(input, kernel_size, filters, stage, block):
    
    # Variables
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # Create layers
    output = keras.layers.Conv2D(filters1, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2a')(input)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(output)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(output)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(output)
    output = keras.layers.add([output, input])
    output = keras.layers.Activation('relu')(output)
    # Return a block
    return output
# Create a convolution block
def conv_block(input, kernel_size, filters, stage, block, strides=(2, 2)):
    # Variables
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # Create block layers
    output = keras.layers.Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '2a')(input)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal', name=conv_name_base + '2b')(output)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(output)
    output = keras.layers.Activation('relu')(output)
    output = keras.layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal', name=conv_name_base + '2c')(output)
    output = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(output)
    shortcut = keras.layers.Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal', name=conv_name_base + '1')(input)
    shortcut = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)
    output = keras.layers.add([output, shortcut])
    output = keras.layers.Activation('relu')(output)
    # Return a block
    return output

# %%
#model=resnet50_model()

#model.layers[-5:]

# %%


# %% [markdown]
# # pretrain ResNet50 on Affwild dataset
# 

# %%
"""

#model = ResNet50( include_top = False)
model_ResNet= ResNet50(outputs=include_top=False, input_shape=(224,224,3), pooling='avg'  )

model=Sequential()
model.add(model_ResNet)
model.add(Dense(2,activation='linear'))
#model.add(Dense(1,activation='linear'))
model.summary()"""

# %% [markdown]
# ### data loading and alignment
# 

# %%
#image_dir = Path('/Users/mac/Downloads/githubdepression/githubdepression/implementation/AV2014_Training_data_NYU/depression_score')
import numpy as np
import pandas as pd
from pathlib import Path
import os.path

#from sklearn.model_selection import train_test_split

#import tensorflow as tf

#from sklearn.metrics import r2_score
#image_dir = Path('/Users/mac/Desktop/implementation_depression/AV2014_Training_data_NYU/depression_score')
#image_dir=Path('/Users/mac/Desktop/implementation_depression/cropped_aligned/4-30-1920x1080')

# %% [markdown]
# # alignment for single folder
# then alignment for all of folders
# 思路：文件夹依次对齐，对每个文件夹里的文件也对齐，对齐过的文件放到想放的路径里，然后，把相应的label，也放进label里面，估计label总计有很多数量。最后直接放server上跑



# %%
import copy
#/scratch/yw4554/im_de_url/implementation_depression/cropped_aligned/scratch/yw4554/im_de_url/implementation_depression/cropped_aligned
training_folder=os.listdir('/scratch/yw4554/im_de_url/implementation_depression/cropped_aligned')
training_folder.sort()
testing_folder=os.listdir('/scratch/yw4554/im_de_url/implementation_depression/Third ABAW Annotations/VA_Estimation_Challenge/merged')
testing_folder.sort()
testing_folder
def remove_suffix(lst):
    for i in range(len(lst)):
        lst[i]=lst[i][:-4]
    return lst
testing_folder=remove_suffix(testing_folder)
training_set=set(training_folder)
testing_set=set(testing_folder)
intersectlst=training_set.intersection(testing_set)
print(len(training_set),len(testing_set),len(intersectlst))
intersectlst_lst=list(intersectlst)
intersectlst_lst.sort()
def add_suffix(lst):
    for i in range(len(lst)):
        lst[i]=lst[i]+'.txt'
    return lst
training_folder=copy.deepcopy(intersectlst_lst)
testing_folder=add_suffix(intersectlst_lst)


def training_folder_prefix():
    for i in range(len(training_folder)):
        training_folder[i]='/scratch/yw4554/im_de_url/implementation_depression/cropped_aligned/'+training_folder[i]
def testing_folder_prefix():
    for j in range(len(testing_folder)):
        testing_folder[j]='/scratch/yw4554/im_de_url/implementation_depression/Third ABAW Annotations/VA_Estimation_Challenge/merged/'+testing_folder[j]
training_folder_prefix()
testing_folder_prefix()

# %%
filepaths=[]
label_scores_lst=[]
def align_single_folder(training_path,testing_path):
    global training2,training,testing
    training2=[]
    training=os.listdir(training_path)

    ####solve the problem
    training_new=[i for i in training if i.endswith('.jpg')]
    training=copy.deepcopy(training_new)
    ##########
    training.sort()
    testing=[]
    with open(testing_path) as f:
        labels=f.readlines()[1:]
    if len(training)<len(labels):
        length=len(training)
    else:
        length=len(labels)
    #length=min(len(training),len(labels))
    for i in range(length):

        inc=int(float(training[i][:-4]))
        if inc<=length:
            #print(inc)
            #print(inc)

            testing.append(labels[inc-1].strip('\n'))
            training2.append(training[i])
    #print(len(training2),len(testing))
    for i in range(len(training2)):
        filepaths.append(training_path+'/'+training2[i])
    for i in range(len(testing)):
        label_scores_lst.append(testing[i])
for i in range(len(training_folder)):
    cur_tr_folder=training_folder[i]
    cur_te_folder=testing_folder[i]
    align_single_folder(cur_tr_folder,cur_te_folder)


    

# %%
len(filepaths)
len(label_scores_lst)

# %%
for i in range(len(label_scores_lst)):
    label_scores_lst[i]=label_scores_lst[i].split(',')

#label_scores_lst

# %%
filepaths=np.asarray(filepaths)
label_V=[]
label_A=[]
for i in range(len(label_scores_lst)):
    label_V.append(label_scores_lst[i][0])
    label_A.append(label_scores_lst[i][1])
label_V=np.asarray(label_V).astype(np.float32)
label_A=np.asarray(label_A).astype(np.float32)

# %%
len(label_A)

# %%


# %%
#filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name='Filepath').astype(str)
V=pd.Series(label_V,name='V')
A=pd.Series(label_A,name='A')
filepaths=pd.Series(filepaths,name='Filepath')
images = pd.concat([filepaths, V,A], axis=1)
import sklearn
from sklearn.model_selection import train_test_split


# %%
images

# %%
image_df = images.sample(100, random_state=1).reset_index(drop=True)
import copy
#image_df = copy.deepcopy(images)
train_df, test_df = train_test_split(image_df, train_size=0.9, shuffle=True, random_state=1)

# %%
train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

# %%
columns=['V','A']
train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col=columns,
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col=columns,
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col=columns,
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='raw',
    batch_size=32,
    shuffle=False
)


# %% [markdown]
# # training

#mirrored_strategy = tf.distribute.MirroredStrategy()
from tensorflow.keras.callbacks import CSVLogger
csv_log = CSVLogger("/scratch/yw4554/im_de_url/implementation_depression/results.csv")
# %%

from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint_path = '/scratch/yw4554/im_de_url/implementation_depression/model_checkpoints/'
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_freq='epoch',
    save_weights_only=True,
    verbose=1
)

#############parallel
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model=resnet50_model()




    model.compile(optimizer='adam',loss='mse')
t1=time.time()
history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=20,
    
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        checkpoint,
        csv_log
    ]
      
     
)
t2=time.time()
print(t2-t1)

# %%
model.save('/scratch/yw4554/im_de_url/implementation_depression/model_big')

# %% [markdown]
# # loading the model

# %%
model2 = keras.models.load_model('/scratch/yw4554/im_de_url/implementation_depression/model_big')

# %%
#model2.summary()

# %% [markdown]
# # remove last layer

# %%
from tensorflow.keras.models import Model
#feature_extraction_model=Model(inputs=model2.input, outputs=model2.layers[-2].output)
feature_extraction_model=Model(inputs=model2.input, outputs=model2.layers[-2].output)
feature_extraction_model.layers[-5:]
#feature_extraction_model.summary()


# %% [markdown]
# # testing

# %%
#predicted_depression = np.squeeze(model.predict(test_images))
#true_depression = test_images.labels

#rmse = np.sqrt(model.evaluate(test_images, verbose=0))
#print("     Test RMSE: {:.5f}".format(rmse))
#from sklearn.metrics import r2_score
#r2 = r2_score(true_depression, predicted_depression)
#print("Test R^2 Score: {:.5f}".format(r2))

# %% [markdown]
# # feature extraction

# %%
import os 
dirs=os.listdir('/scratch/yw4554/im_de_url/implementation_depression/20-50/train/20')
paths=[]
for i in range(len(dirs)):
    paths.append('/scratch/yw4554/im_de_url/implementation_depression/20-50/train/20/'+dirs[i])
paths=paths[:50]
#paths

# %%
biglst=[]
for i in range(len(paths)):
    orig = cv.imread(paths[i])

    # Convert image to RGB from BGR (another way is to use "image = image[:, :, ::-1]" code)
    orig = cv.cvtColor(orig, cv.COLOR_BGR2RGB)

    # Resize image to 224x224 size
    image = cv.resize(orig, (224, 224)).reshape(-1, 224, 224, 3)

    # We need to preprocess imageto fulfill ResNet50 requirements
    image = preprocess_input(image)

    # Extracting our features
    features = feature_extraction_model.predict(image)
    toaddlst=np.ndarray.tolist(features[0])
    biglst.append(toaddlst)


    features.shape

# %%
features.shape

# %%
features[0]

# %%
#features[0][0][0]
#for i in range(len(features[0][0][0])):
#   print(features[0][0][0][i])

# %%

field=['name','timestamp']
for i in range(2048):
    field.append('neuron_'+str(i))

# %%
import csv
video_name='203_1_Freeform_video'
with open("/scratch/yw4554/im_de_url/implementation_depression/feature_ResNet50_epoch_10.csv",'w') as f:
    
    write = csv.writer(f)
    write.writerow(field)

   
    for i in range(len(biglst)):
        

        newlst=[video_name,str(i)]
        
        newlst+=biglst[i]
            
      
        write.writerow(newlst)
    
f.close()

print(t2-t1)
            
        

        

# %%
#!scp -r /Users/mac/Desktop/implementation_depression/cropped_aligned yw4554@greene.hpc.nyu.edu:/scratch/yw4554/implementation_depression


