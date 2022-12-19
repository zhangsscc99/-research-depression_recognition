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
from keras.layers import *
from tensorflow.keras.models import Model

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

base_model = ResNet50(weights=None, include_top=False)

# Add a new top layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
#x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='linear')(x)

# Create a new model



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

# %%
#model=resnet50_model()


# %% [markdown]
# ### data loading and alignment
# 

# %%
#image_dir = Path('/Users/mac/Downloads/githubdepression/githubdepression/implementation/AV2014_Training_data_NYU/depression_score')
import numpy as np
import pandas as pd
from pathlib import Path
import os.path



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
filepaths=filepaths[:100000]
label_scores_lst=label_scores_lst[:100000]

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
#image_df = images.sample(100, random_state=1).reset_index(drop=True)
import copy
image_df = copy.deepcopy(images)
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



model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam',loss='mse')
t1=time.time()
history = model.fit(
    train_images,
    validation_data=val_images,
    epochs=20,
    
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
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

model3=ResNet50(weights=None, include_top=False)
feature_extraction_model2=Model(inputs=model3.input, outputs=model3.layers[-2].output)


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
with open("/scratch/yw4554/im_de_url/implementation_depression/feature_ResNet50_epoch_20.csv",'w') as f:
    
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
    features = feature_extraction_model2.predict(image)
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
with open("/scratch/yw4554/im_de_url/implementation_depression/feature_ResNet50_feature2.csv",'w') as f:
    
    write = csv.writer(f)
    write.writerow(field)

   
    for i in range(len(biglst)):
        

        newlst=[video_name,str(i)]
        
        newlst+=biglst[i]
            
      
        write.writerow(newlst)
    
f.close()


