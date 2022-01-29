import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle


import warnings

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Permute
from tensorflow.keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard

import tensorflow as tf

import pdb

from custom_generator import CustomDataGen

directory=['/home/julian/lidar_image/motion_images/', '/home/julian/lidar_image/motion_images2/', '/home/julian/lidar_image/motion_images3/']
#directory=['/home/julian/lidar_image/motion_images2/']


mpl.style.use('seaborn-paper')
warnings.simplefilter('ignore', category=DeprecationWarning)


if not os.path.exists('weights/'):
    os.makedirs('weights/')


def append_ext(fn):
    return fn+".eps.png"


def load_dataset_at(df, evaluation=False):

    batch_size = 64
    image_size = (400,400)
    rescale = 1./65535


    if(evaluation):

        datagen = ImageDataGenerator(rescale=1./65535)

        test_generator = datagen.flow_from_dataframe(
        	    df,
            	batch_size=batch_size,
	            target_size=image_size,
        	    color_mode="grayscale", #"grayscale",
    	        class_mode="raw",
            	shuffle=False,
	            valid_filenames=False
        )

        return test_generator


    else:
        datagen = ImageDataGenerator(rescale=1./65535, validation_split=0.1, width_shift_range=0.05, height_shift_range=0.05, zoom_range=[0.75, 1.25])

        train_generator = datagen.flow_from_dataframe(
        	    df,
            	batch_size=batch_size,
	            target_size=image_size,
            	color_mode="grayscale", #"grayscale",
	            class_mode="raw",
            	subset="training",
	            valid_filenames=False
        )
        

        valid_generator = datagen.flow_from_dataframe(
            	df,
        	    batch_size=batch_size,
    	        target_size=image_size,
            	color_mode="grayscale", #"grayscale",
	            class_mode="raw",
        	    subset="validation",
    	        valid_filenames=False
        )
   

    return train_generator, valid_generator


def load_dataset_at2(traindf, valdf, evaluation=False, frames=1):

    batch_size = 64
    image_size = (400,400)
    rescale = 1./65535


    if evaluation:

        train_generator = CustomDataGen(traindf, batch_size=batch_size, frames=frames, rescale=rescale, shuffle=False, augment=False)
        valid_generator = ""

    else:
        train_generator = CustomDataGen(traindf, batch_size=batch_size, frames=frames, rescale=rescale, augment=True)
        valid_generator = CustomDataGen(valdf, batch_size=batch_size, frames=frames, rescale=rescale, augment=True)

    return train_generator, valid_generator

def train_model(model: Model, epochs=50, batch_size=128, learning_rate=1e-3, frames=1):

    #pdb.set_trace()
    if frames == 1:
        tmpdf = []
        for f in directory:
            df = pd.read_csv(f + "train.csv", dtype={'prefix': str, 'filename': str, 'class': np.float64})
            df["filename"] = f + df["prefix"] + "/" + df["filename"]
            df["filename"]=df["filename"].apply(append_ext)
            tmpdf.append(df)

        traindf = pd.concat(tmpdf, ignore_index=True)
                
    else:
        dtype_dict = {'prefix': str, 'class': np.float64}
        
        #for f in range(frames):
        #filename_str = "filename" + f
        filename_strs = ["filename" + str(f) for f in range(frames)]
        dtype_dict = dict.fromkeys(filename_strs, str)
        tmpdf1 = []
        tmpdf2 = []
        #pdb.set_trace()

        for f in directory:
            df1 = pd.read_csv(f + "train" + str(frames) + ".csv", dtype=dtype_dict)
            df2 = pd.read_csv(f + "val" + str(frames) + ".csv", dtype=dtype_dict)
            df1["prefix"] = f + df1["prefix"].astype(str)
            df2["prefix"] = f + df2["prefix"].astype(str)
            for t in filename_strs:
                df1[t]=df1[t].apply(append_ext)
                df2[t]=df2[t].apply(append_ext)

            tmpdf1.append(df1)
            tmpdf2.append(df2)

        traindf = pd.concat(tmpdf1, ignore_index=True)
        valdf = pd.concat(tmpdf2, ignore_index=True)
        """
        traindf = pd.read_csv(directory + "train" + str(frames) + ".csv", dtype=dtype_dict)
        valdf = pd.read_csv(directory + "val" + str(frames) + ".csv", dtype=dtype_dict)
        for f in filename_strs:
            traindf[f]=traindf[f].apply(append_ext)
            valdf[f]=valdf[f].apply(append_ext)
        """
            


        

    if frames == 1:
        g_train, g_valid = load_dataset_at(traindf, evaluation=False)
    else:
        g_train, g_valid = load_dataset_at2(traindf, valdf, evaluation=False, frames=frames)

    factor = 1. / np.cbrt(2)    

    model_checkpoint = ModelCheckpoint("./weights/_weights.h5", verbose=1,
                                       monitor='val_loss', save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', factor=factor,
                                  cooldown=0, min_lr=1e-4, verbose=2)

    tensorboard_callback = TensorBoard(log_dir="logs", histogram_freq=1)

    callback_list = [model_checkpoint, reduce_lr, tensorboard_callback] #, reduce_lr]

    optm = Adam(lr=learning_rate)

    model.compile(optimizer='adam', loss='mean_squared_error')
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    STEP_SIZE_TRAIN = g_train.n//g_train.batch_size
    STEP_SIZE_VALID = g_valid.n//g_valid.batch_size

    #pdb.set_trace()
    model.fit(g_train, steps_per_epoch=STEP_SIZE_TRAIN, epochs=epochs, callbacks=callback_list, verbose=1, validation_data=g_valid, validation_steps=STEP_SIZE_VALID)
    #model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list, verbose=2, validation_data=(X_test, y_test))



def evaluate_model(model: Model, batch_size=128, frames=1):

    #pdb.set_trace()
    if frames == 1:
        tmpdf = []
        for f in directory:
            df = pd.read_csv(f + "test.csv", dtype={'prefix': str, 'filename': str, 'class': np.float64})
            df["filename"] = f + df["prefix"] + "/" + df["filename"]
            df["filename"]=df["filename"].apply(append_ext)
            tmpdf.append(df)

        testdf = pd.concat(tmpdf, ignore_index=True)
                
    else:
        dtype_dict = {'prefix': str, 'class': np.float64}
        
        #for f in range(frames):
        #filename_str = "filename" + f
        filename_strs = ["filename" + str(f) for f in range(frames)]
        dtype_dict = dict.fromkeys(filename_strs, str)
        tmpdf1 = []
        tmpdf2 = []
        #pdb.set_trace()

        for f in directory:
            df1 = pd.read_csv(f + "test" + str(frames) + ".csv", dtype=dtype_dict)
            df1["prefix"] = f + df1["prefix"].astype(str)
            for t in filename_strs:
                df1[t]=df1[t].apply(append_ext)

            tmpdf1.append(df1)

        testdf = pd.concat(tmpdf1, ignore_index=True)


    """
    if frames == 1:
        testdf = pd.read_csv(directory + "test.csv", dtype={'prefix': str, 'filename': str, 'class': np.float64})
        testdf["filename"] = testdf["prefix"] + "/" + testdf["filename"]
        testdf["filename"]=testdf["filename"].apply(append_ext)
    else:
        dtype_dict = {'prefix': str, 'class': np.float64}
        
        #for f in range(frames):
        #filename_str = "filename" + f
        filename_strs = ["filename" + str(f) for f in range(frames)]
        dtype_dict = dict.fromkeys(filename_strs, str)
        testdf = pd.read_csv(directory + "test" + str(frames) + ".csv", dtype=dtype_dict)
        for f in filename_strs:
            testdf[f]=testdf[f].apply(append_ext)
    """

    if frames == 1:
        g_test = load_dataset_at(testdf, evaluation=True)
    else:
        g_test,_ = load_dataset_at2(testdf, [], evaluation=True, frames=frames)

                                                          

    optm = Adam(lr=1e-3)
    model.compile(optimizer=optm, loss='mean_absolute_error')
    #model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])

    model.load_weights("./weights/_weights.h5")
    print("Weights loaded from ", "./weights/_weights.h5")

    STEP_SIZE_TEST = g_test.n//g_test.batch_size

    #pdb.set_trace()
    print("\nEvaluating : ")
    loss = model.evaluate(g_test, steps=STEP_SIZE_TEST)
    #loss, accuracy= model.evaluate(np.expand_dims(X_test,1), y_test, batch_size=batch_size)
    print()
    print("Final loss : ", loss)
    #print("Final accuracy : ", accuracy)

    results = model.predict(g_test) #, steps=STEP_SIZE_TEST)

    true_labels = g_test.labels
    #pdb.set_trace()
    """
    results_end = np.argmax(results, axis=1)*10+100
    plt.hist(results_end, bins=list(range(100,510,10)))
    plt.title("Frequency of predictions: " + str(NUM_FRAMES) + " channels")
    plt.xlabel('Frequency (Hz)')
    plt.show()
    print(int(np.sum(y_test_tmp == 100)))
    #plt.hist(y_test_tmp, bins=list(range(100,510,10)))
    #plt.show()
    """
    """
    plt.hist(results[:,0]-y_test, bins=20)
    plt.xlabel('Error (Hz)')
    plt.title("Error distribution: " + str(NUM_FRAMES) + " channels")
    plt.show()
    """

    number_freqs = 41
    #f_elements = int(np.sum(y_test == 100))
    #frequency_dist = np.zeros((5, f_elements))
    #pdb.set_trace()
    number_instances, edges = np.histogram(true_labels, 'auto', range=(true_labels.min()-1, true_labels.max()+1))
    number_freqs = len(number_instances)
    frequencies = np.zeros((number_freqs))

    label_inds = np.digitize(true_labels, edges)
    frequency_dist = {}
    frequency_dist_true = {}
    for idx,s in enumerate(true_labels):
        if idx >= len(results):
            break
        frequencies[label_inds[idx]-1] += np.abs(results[idx,0] - s)
        if label_inds[idx] % 1 == 0:
            if label_inds[idx] not in frequency_dist:
                frequency_dist[label_inds[idx]] = np.array([])
                frequency_dist_true[label_inds[idx]] = np.array([])
            frequency_dist[label_inds[idx]] = np.append(frequency_dist[label_inds[idx]],results[idx][0])
            frequency_dist_true[label_inds[idx]] = np.append(frequency_dist_true[label_inds[idx]],s)

        
    x_values = np.zeros((number_freqs))
    for idx in range(len(edges)):
        
        if idx > 0:
            x_values[idx-1] = np.mean([edges[idx-1], edges[idx]])
    
        #number_instances = np.sum(label_inds == idx+1)
        if idx < len(number_instances) and number_instances[idx] > 0:
            frequencies[idx] /= number_instances[idx]

    #pdb.set_trace()
    fig = plt.figure(1)
    subplot_num = 1
    number_real_freqs = len(frequency_dist.keys())
    #for idx in range(5):
    for freq,arr in frequency_dist.items():
        plt.subplot(number_real_freqs, 1, subplot_num)
        plt.hist(arr, density=True, bins=10)
        #plt.hist(frequency_dist_true[freq], density=True, bins=20)
        subplot_num += 1
        plt.title("Error distribution for " + str(x_values[freq-1]) +  " RPM")
        
    plt.xlabel('RPM')
    fig.tight_layout(pad=2.0)
    plt.show()
    pickle.dump(fig, open('frequency_dist.fig.pickle', 'wb'))





    #pdb.set_trace()
    zero_index = np.where(frequencies == 0)
    x_values = np.delete(x_values, zero_index)
    frequencies = np.delete(frequencies, zero_index)
    fig = plt.figure(2)
    plt.plot(x_values, frequencies)
    
    plt.xlabel('Rotational Velocity (RPM)')
    plt.ylabel('Mean absolute error (RPM)')
    plt.title("Mean absolute error for each velocity")
    plt.show()
    pickle.dump(fig, open('rpm_dist.fig.pickle', 'wb'))

    
    
    
    """
    for idx,r in enumerate(results):
        print(y_test[idx], ",", r[0])
    #print(results)
    #print(y_test)
    """
    return loss



