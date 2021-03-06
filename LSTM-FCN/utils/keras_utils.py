import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime


import warnings

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Permute
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard


from tensorflow.keras.utils import to_categorical

from utils.generic_utils import load_dataset_at
from utils.constants import MAX_SEQUENCE_LENGTH_LIST, TRAIN_FILES

import matplotlib.pyplot as plt

import tensorflow as tf

import pdb
import pickle

mpl.style.use('seaborn-paper')
warnings.simplefilter('ignore', category=DeprecationWarning)


if not os.path.exists('weights/'):
    os.makedirs('weights/')


def train_model(model: Model, dataset_id, dataset_prefix, NUM_FRAMES, MAX_SEQUENCE_LENGTH, epochs=50, batch_size=128, val_subset=None,
                cutoff=None, normalize_timeseries=False, learning_rate=1e-3, reload=False):
    """
    Trains a provided Model, given a dataset id.

    Args:
        model: A Keras Model.
        dataset_id: Integer id representing the dataset index containd in
            `utils/constants.py`.
        dataset_prefix: Name of the dataset. Used for weight saving.
        epochs: Number of epochs to train.
        batch_size: Size of each batch for training.
        val_subset: Optional integer id to subset the test set. To be used if
            the test set evaluation time significantly surpasses training time
            per epoch.
        cutoff: Optional integer which slices of the first `cutoff` timesteps
            from the input signal.
        normalize_timeseries: Bool / Integer. Determines whether to normalize
            the timeseries.

            If False, does not normalize the time series.
            If True / int not equal to 2, performs standard sample-wise
                z-normalization.
            If 2: Performs full dataset z-normalization.
        learning_rate: Initial learning rate.
    """
    X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(dataset_id, NUM_FRAMES, 	      MAX_SEQUENCE_LENGTH, normalize_timeseries=normalize_timeseries)
    #pdb.set_trace()
    #max_nb_words, sequence_length = calculate_dataset_metrics(X_train)

    """
    if sequence_length != MAX_SEQUENCE_LENGTH_LIST[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, sequence_length)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            X_train, X_test = cutoff_sequence(X_train, X_test, choice, dataset_id, sequence_length)

    if not is_timeseries:
        X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH_LIST[dataset_id], padding='post', truncating='post')
        X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH_LIST[dataset_id], padding='post', truncating='post')

    classes = np.unique(y_train)
    le = LabelEncoder()
    y_ind = le.fit_transform(y_train.ravel())
    recip_freq = len(y_train) / (len(le.classes_) *
                                 np.bincount(y_ind).astype(np.float64))
    class_weight = recip_freq[le.transform(classes)]

    print("Class weights : ", class_weight)
    """
    
    """
    #pdb.set_trace()
    y_train = (y_train-100)/10
    y_test = (y_test-100)/10
    y_train = to_categorical(y_train, len(np.unique(y_train)))
    y_test = to_categorical(y_test, len(np.unique(y_test)))
    """
    
    
    if is_timeseries:
        factor = 1. / np.cbrt(2)
    else:
        factor = 1. / np.sqrt(2)
    

    path_splits = os.path.split(dataset_prefix)
    if len(path_splits) > 1:
        base_path = os.path.join('weights', *path_splits)

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        base_path = os.path.join(base_path, path_splits[-1])

    else:
        all_weights_path = os.path.join('weights', dataset_prefix)

        if not os.path.exists(all_weights_path):
            os.makedirs(all_weights_path)

    model_checkpoint = ModelCheckpoint("./weights/%s_weights.h5" % dataset_prefix, verbose=1,
                                       monitor='val_loss', save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto',
                                  factor=factor, cooldown=0, min_lr=1e-4, verbose=2)

    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    callback_list = [model_checkpoint, reduce_lr, tensorboard_callback] #, reduce_lr]

    optm = Adam(lr=learning_rate)

    model.compile(optimizer='adam', loss='mean_squared_error')
 
    if(reload):
        model.load_weights("./weights/_weights.h5")

    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    #pdb.set_trace()
    model.fit(np.expand_dims(X_train,1), y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list, verbose=1, validation_data=(np.expand_dims(X_test,1), y_test))
    #model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callback_list, verbose=2, validation_data=(X_test, y_test))



def evaluate_model(model: Model, dataset_id, dataset_prefix, NUM_FRAMES, MAX_SEQUENCE_LENGTH, batch_size=128, test_data_subset=None,
                   cutoff=None, normalize_timeseries=False):
    """
    Evaluates a given Keras Model on the provided dataset.

    Args:
        model: A Keras Model.
        dataset_id: Integer id representing the dataset index containd in
            `utils/constants.py`.
        dataset_prefix: Name of the dataset. Used for weight saving.
        batch_size: Size of each batch for evaluation.
        test_data_subset: Optional integer id to subset the test set. To be used if
            the test set evaluation time is significantly.
        cutoff: Optional integer which slices of the first `cutoff` timesteps
            from the input signal.
        normalize_timeseries: Bool / Integer. Determines whether to normalize
            the timeseries.

            If False, does not normalize the time series.
            If True / int not equal to 2, performs standard sample-wise
                z-normalization.
            If 2: Performs full dataset z-normalization.

    Returns:
        The test set accuracy of the model.
    """
    _, _, X_test, y_test, is_timeseries = load_dataset_at(dataset_id, NUM_FRAMES, MAX_SEQUENCE_LENGTH, normalize_timeseries=normalize_timeseries, evaluation=True)
                                                          
    """
   
    max_nb_words, sequence_length = calculate_dataset_metrics(X_test)

    if sequence_length != MAX_SEQUENCE_LENGTH_LIST[dataset_id]:
        if cutoff is None:
            choice = cutoff_choice(dataset_id, sequence_length)
        else:
            assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
            choice = cutoff

        if choice not in ['pre', 'post']:
            return
        else:
            _, X_test = cutoff_sequence(None, X_test, choice, dataset_id, sequence_length)

    if not is_timeseries:
        X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH_LIST[dataset_id], padding='post', truncating='post')
    """
    """
    y_test_tmp = y_test
    y_test = (y_test-100)/10
    y_test = to_categorical(y_test, len(np.unique(y_test)))
    """
    

    optm = Adam(lr=1e-3)
    model.compile(optimizer=optm, loss='mean_absolute_error')
    #model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
    #pdb.set_trace()
    model.load_weights("./weights/_weights.h5")
    print("Weights loaded from ", "./weights/%s_weights.h5" % dataset_prefix)

    
    #pdb.set_trace()
    print("\nEvaluating : ")
    loss = model.evaluate(np.expand_dims(X_test,1), y_test, batch_size=batch_size)
    #loss, accuracy= model.evaluate(np.expand_dims(X_test,1), y_test, batch_size=batch_size)
    print()
    print("Final loss : ", loss)
    #print("Final accuracy : ", accuracy)

    results = model.predict(np.expand_dims(X_test,1), batch_size=batch_size)
    
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
    
    #pdb.set_trace()
    y_test, results = (np.array(t) for t in zip(*sorted(zip(y_test, results))))
    unique_elements = np.unique(y_test)
    frequencies = np.zeros((unique_elements.size)) #41
    f_sizes = {}
    for i in unique_elements:
        f_sizes[i] = np.count_nonzero(y_test == i)

    #f_elements = int(np.sum(y_test == 100))
    #frequency_dist_num = 4
    frequency_dist = {} #np.zeros((frequency_dist_num, np.max(f_sizes)))
    #pdb.set_trace()
    past_freq = y_test[0]
    #count_freq = []
    for idx,s in enumerate(y_test):
        #print(s, np.abs(results[idx][0] - s))
        frequencies[int((s)/10-1)] += np.abs(results[idx][0] - s)
        if s % 50 == 0:
            if past_freq != s:
                past_freq = s
                #count_freq = []
                frequency_dist[s] = np.array([])
            frequency_dist[s] = np.append(frequency_dist[s],np.abs(results[idx][0]-s))
            #frequency_dist[int(s/50)-1, count_freq] = results[idx][0]
            
            #count_freq += 1
        
    #pdb.set_trace()

    for idx in range(frequencies.size):
        frequencies[idx] /= f_sizes[idx*10+10] #np.sum(y_test == 100)
    #pdb.set_trace()
    fig = plt.figure(1)
    subplot_num = 811
    #for idx in range(5):
    for freq,arr in frequency_dist.items():
        plt.subplot(subplot_num)
        #plt.hist(arr, density=True, bins=20)
        plt.plot(arr)
        subplot_num += 1
        plt.title("Error distribution for " + str(freq) +  "Hz")
        
    plt.xlabel('Frequency (Hz)')
    fig.tight_layout(pad=2.0)
    plt.show()
    pickle.dump(fig, open('frequency_dist.fig.pickle', 'wb'))
    #plt.savefig('frequency_dist.png')
    
    
    fig = plt.figure(2)   
    #plt.plot(list(range(100,510,10)), frequencies)
    plt.plot(list(range(10,410,10)), frequencies)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mean absolute error (Hz)')
    plt.title("Mean absolute error for each frequency at: " + str(NUM_FRAMES) + " channels")
    plt.show()
    pickle.dump(fig, open('frequency_dist2.fig.pickle', 'wb'))
    #plt.savefig('frequency_dist2.png')

    
    
    
    """
    for idx,r in enumerate(results):
        print(y_test[idx], ",", r[0])
    #print(results)
    #print(y_test)
    """
    return loss



