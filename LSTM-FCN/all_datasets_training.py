import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from tensorflow.python.keras.backend import set_session
config = tf.compat.v1.ConfigProto() 
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess2 = tf.compat.v1.Session(config=config)
set_session(sess2)  # set this TensorFlow session as the default

from tensorflow.keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Flatten, Lambda
from tensorflow.keras.layers import Input, Dense, LSTM, concatenate, Activation, GRU, SimpleRNN, MaxPool1D #, CuDNNLSTM
from tensorflow.keras.models import Model

from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from utils.keras_utils import train_model, evaluate_model
#from utils.layer_utils import AttentionLSTM
from tensorflow.signal import rfft

import tensorflow as tf
import sys

import pdb

"""
Change from linear regression to classification: check generate_lstmfcn, change NB_CLASS, check train_model
"""

#380 + 96
NUM_FRAMES = 14 #30#14#15#5#75
MAX_SEQUENCE_LENGTH = 20 # 20 #80 #70 #MAX_SEQUENCE_LENGTH_LIST[did]


def generate_lstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    
    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH*NUM_FRAMES))

    y = Permute((2, 1))(ip)
    
    kernel_size = [100,200, 400, 200, 100] #[100,200,400,800]#[100,200, 400, 200, 100]
    kernel = 15 #15
 
    y = Lambda(lambda x: tf.math.abs(rfft(x)), name = "Lambda_rfft1")(y)

    for k in kernel_size:
        y = Conv1D(k, kernel, padding='same', kernel_initializer='he_uniform')(y)
        y = MaxPool1D()(y) 
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
        y = Dropout(0.1)(y)

  
    y = LSTM(10)(y)
    y = Dropout(0.5)(y)

    #pdb.set_trace()
    out = Dense(NB_CLASS, activation='linear')(y)
    #out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model

"""
def generate_lstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    
    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH*NUM_FRAMES))

    #pdb.set_trace()
    
    x = LSTM(NUM_CELLS)(ip)

    x = Dropout(0.8)(x)
    
    
    #x = Lambda(lambda x: tf.expand_dims(x,1))(x)
    
    #x = Lambda(lambda x: tf.math.abs(rfft(x)), name = "Lambda_rfft1")(x)
    
    #y = Lambda(lambda x: tf.math.abs(rfft(x)), name = "Lambda_rfft1")(ip)
    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)
    
    x = concatenate([x, y])
    
    

    out = Dense(NB_CLASS, activation='linear')(x)
    #out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model

"""

"""
def generate_alstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = AttentionLSTM(NUM_CELLS)(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='linear')(x)
    #out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model
"""

if __name__ == "__main__":


    eval_mode = False
    reload = False
    if len(sys.argv) > 1:
        if int(sys.argv[1]) == 1:
            eval_mode = True
        else:
            reload = True
        
    base_log_name = '%s_%d_cells_new_datasets.csv'
    base_weights_dir = '%s_%d_cells_weights/'

  

    # Normalization scheme
    # Normalize = False means no normalization will be done
    # Normalize = True / 1 means sample wise z-normalization
    # Normalize = 2 means dataset normalization.
    normalize_dataset = False

    #for model_id, (MODEL_NAME, model_fn) in enumerate(MODELS):
    model_id = 0
    MODEL_NAME = 'lstmfcn'
    model_fn = generate_lstmfcn
    cell = 64
        
        #if model_id == 0:
        #    continue
        
        #for cell in CELLS:

    successes = []
    failures = []

    if not os.path.exists(base_log_name % (MODEL_NAME, cell)):
        file = open(base_log_name % (MODEL_NAME, cell), 'w')
        file.write('%s,%s,%s,%s\n' % ('dataset_id', 'dataset_name', 'dataset_name_', 'test_accuracy'))
        file.close()

    #for dname, did in dataset_map:

    NB_CLASS = 1#1 #41#NB_CLASSES_LIST[did]

    # release GPU Memory


    file = open(base_log_name % (MODEL_NAME, cell), 'a+')

    weights_dir = base_weights_dir % (MODEL_NAME, cell)

    if not os.path.exists('weights/' + weights_dir):
        os.makedirs('weights/' + weights_dir)

    dataset_name_ = ""
    dname = ""
    did = 1

    # try:
    model = model_fn(MAX_SEQUENCE_LENGTH, NB_CLASS, cell)

    print('*' * 20, "Training model for dataset %s" % (dname), '*' * 20)

    # comment out the training code to only evaluate !
    #pdb.set_trace()
    if not eval_mode:
        train_model(model, did, dataset_name_, NUM_FRAMES=NUM_FRAMES, MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH, epochs=2000, batch_size=64, normalize_timeseries=normalize_dataset, reload=reload)

    
    acc = evaluate_model(model, did, dataset_name_, batch_size=64,
                         normalize_timeseries=normalize_dataset, NUM_FRAMES=NUM_FRAMES,            MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH)
                         
    
    
    """
    s = "%d,%s,%s,%0.6f\n" % (did, dname, dataset_name_, acc)
    
    file.write(s)
    file.flush()

    successes.append(s)

            # except Exception as e:
            #     traceback.print_exc()
            #
            #     s = "%d,%s,%s,%s\n" % (did, dname, dataset_name_, 0.0)
            #     failures.append(s)
            #
            #     print()

            file.close()
            
            print('\n\n')
            print('*' * 20, "Successes", '*' * 20)
            print()

            for line in successes:
                print(line)

            print('\n\n')
            print('*' * 20, "Failures", '*' * 20)
            print()

            for line in failures:
                print(line)
    """
