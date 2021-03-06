import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pylab as plt
import pdb
from tqdm import tqdm

mpl.style.use('seaborn-paper')

from utils.constants import TRAIN_FILES, TEST_FILES, MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST

freq_dirs = ["../data_4m3"] #["../data_4m2", "../data_4m3",  "../data_2m", "../data_3m2", "../data4"] #, "../data_low"] #["../data_non"] #["../data_4m", "../data_3m"] #["../data_lower"] #["../data_low", "../data4"] #["../data4"] #["../data3", "../data4"]


"""
def get_data(split):

    NUM_SAMPLES = 80 #70
    #NUM_CHANNELS = 16 #14
    first = True
    NUM_FRAMES = 30
    

    freq_dir = "../data2"
    NUM_FILES = len(os.listdir(freq_dir + '/' + str(100) + '/' + split))
    final_time_series = np.array([])
    final_time_labels = np.array([])
    
    #for i in range(100, 550, 50):

    for i in range(100,510,10):
        residue = 0
        tmp_time_series = []
        for j in range(1,NUM_FILES+1):
            freq_path = os.path.join(freq_dir, str(i), split, str(j) + ".csv")
            time_series = pd.read_csv(freq_path, header=None).to_numpy()
            time_series = time_series[:,:NUM_SAMPLES]
            NUM_CHANNELS = time_series.shape[0]

           
            for k in range(NUM_CHANNELS):
                if residue < NUM_FRAMES:
                    tmp_time_series = np.concatenate((tmp_time_series, time_series[k,:]))
                    residue += 1
                else:
                    if not first:

                        final_time_series = np.concatenate((final_time_series, np.expand_dims(tmp_time_series,0)))
                        final_time_labels = np.concatenate((final_time_labels, [i]))
                    else:
                        final_time_series = np.expand_dims(tmp_time_series,0)
                        final_time_labels = np.array([i])
                        first = False
                    residue = 0
                    tmp_time_series = []
            

            
            

    return final_time_series, final_time_labels

"""
"""
def get_data(split, NUM_FRAMES, MAX_SEQUENCE_LENGTH):

    NUM_SAMPLES = MAX_SEQUENCE_LENGTH #40 #80 #70
    #NUM_CHANNELS = 16 #14
    first = True
    #NUM_FRAMES = 30
    real_size = NUM_SAMPLES*NUM_FRAMES 

    #pdb.set_trace()
    final_time_series = np.array([])
    final_time_labels = np.array([])

    for freq_dir in freq_dirs:
        NUM_FILES = len(os.listdir(freq_dir + '/' + str(100) + '/' + split))
        for i in tqdm(range(10,410,10)):#510,10)):
            residue = []
            for j in range(1,NUM_FILES+1):
                freq_path = os.path.join(freq_dir, str(i), split, str(j) + ".csv")
                time_series = pd.read_csv(freq_path, header=None).to_numpy().flatten()
                #time_series_r = time_series[:,NUM_SAMPLES:]
                #pdb.set_trace()
                time_series = np.concatenate((residue, time_series))
                NUM_SEQUENCES = len(time_series)//(real_size)
                
                if len(time_series) % real_size > 0:
                    residue = time_series[NUM_SEQUENCES*real_size:]
                else:
                    residue = []

                if NUM_SEQUENCES > 0:
                    tmp_time_series = np.resize(time_series,(NUM_SEQUENCES, real_size))

                    if not first:

                        final_time_series = np.concatenate((final_time_series, tmp_time_series)) #np.expand_dims(time_series,0)))
                        final_time_labels = np.concatenate((final_time_labels, [i]*(NUM_SEQUENCES)))
                    else:
                        final_time_series = tmp_time_series #np.expand_dims(time_series,0)
                        final_time_labels = np.array([i]*(NUM_SEQUENCES))
                        first = False
                

                
            
    #pdb.set_trace()
    print('Samples size:', final_time_series.shape)
    return final_time_series, final_time_labels

"""

def get_data(split, NUM_FRAMES, MAX_SEQUENCE_LENGTH):

    NUM_SAMPLES = MAX_SEQUENCE_LENGTH #40 #80 #70
    #NUM_CHANNELS = 16 #14
    first = True
    #NUM_FRAMES = 30
    


    final_time_series = np.array([])
    final_time_labels = np.array([])
    
    #for i in range(100, 550, 50):
    for freq_dir in freq_dirs:
        NUM_FILES = len(os.listdir(freq_dir + '/' + str(100) + '/' + split))
        for i in tqdm(range(10,410,10)):#510,10)):
            residue = 0
            tmp_time_series = []
            for j in range(1,NUM_FILES+1):
                freq_path = os.path.join(freq_dir, str(i), split, str(j) + ".csv")
                time_series = pd.read_csv(freq_path, header=None).to_numpy()
                #time_series_r = time_series[:,NUM_SAMPLES:]
                #pdb.set_trace()
                if time_series.shape[1] < NUM_SAMPLES:
                    num_frames = len(time_series.flatten())//NUM_SAMPLES
                    time_series = np.resize(time_series,(num_frames,NUM_SAMPLES))
                time_series = time_series[:,:NUM_SAMPLES]
                NUM_CHANNELS = time_series.shape[0]

               
                for k in range(NUM_CHANNELS):
                    if residue < NUM_FRAMES:
                        tmp_time_series = np.concatenate((tmp_time_series, time_series[k,:]))
                        residue += 1
                    else:
                        if not first:

                            final_time_series = np.concatenate((final_time_series, np.expand_dims(tmp_time_series,0)))
                            final_time_labels = np.concatenate((final_time_labels, [i]))
                        else:
                            final_time_series = np.expand_dims(tmp_time_series,0)
                            final_time_labels = np.array([i])
                            first = False
                        residue = 0
                        tmp_time_series = []
                

                
            
    #pdb.set_trace()
    print('Samples size:', final_time_series.shape)
    return final_time_series, final_time_labels




"""
def get_data(split, NUM_FRAMES):

    NUM_SAMPLES = 80#89 #70
    NUM_CHANNELS = 15#16 #14
    first = True
    
    freq_dirs = ["../data4"] #["../data3", "../data4"]
        

    final_time_series = np.array([])
    final_time_labels = np.array([])
    tmp_time_series = np.array([])
    CONCAT_AXIS = 1
    
    for freq_dir in freq_dirs:
        NUM_FILES = len(os.listdir(freq_dir + '/' + str(100) + '/' + split))
        EFFECTIVE_FILES = NUM_FILES - NUM_FILES % NUM_FRAMES

        for i in tqdm(range(100,510,10)):
            for j in range(1,NUM_FILES+1):
                freq_path = os.path.join(freq_dir, str(i), split, str(j) + ".csv")
                time_series = pd.read_csv(freq_path, header=None).to_numpy()
                time_series = time_series[:NUM_CHANNELS,:NUM_SAMPLES]
                #pdb.set_trace()
                
                if NUM_FRAMES > 1:
                    if not first:
                    
                        if j % NUM_FRAMES == 1:
                            tmp_time_series = time_series
                            final_time_labels = np.concatenate((final_time_labels, [i]*NUM_CHANNELS))
                        else:
                            tmp_time_series = np.concatenate((tmp_time_series, time_series),CONCAT_AXIS)
                        if j % NUM_FRAMES == 0:
                            final_time_series = np.concatenate((final_time_series, tmp_time_series))
                    else:
                        
                        if j % NUM_FRAMES == 1:
                            tmp_time_series = time_series
                            final_time_labels = np.array([i]*NUM_CHANNELS)
                        else:
                            tmp_time_series = np.concatenate((tmp_time_series, time_series),CONCAT_AXIS)
                        if j % NUM_FRAMES == 0:
                            final_time_series = tmp_time_series #np.expand_dims(tmp_time_series, 0)
                            first = False
                    if j == EFFECTIVE_FILES:
                        break
                else:
                    if not first:
                        final_time_series = np.concatenate((final_time_series, [time_series]))
                        final_time_labels = np.concatenate((final_time_labels, [i]))
                    else:
                        final_time_series = np.expand_dims(time_series,0)
                        final_time_labels = np.array([i])
                        first = False

    #pdb.set_trace()
    print('Samples size:', final_time_series.shape)
    return final_time_series, final_time_labels

"""
"""
def get_data(split):

    NUM_SAMPLES = 89 #70
    #NUM_CHANNELS = 16 #14
    first = True
    
    #pdb.set_trace()
    freq_dir = "../data3"
    NUM_FILES = len(os.listdir(freq_dir + '/' + str(100) + '/' + split))
    final_time_series = np.array([])
    final_time_labels = np.array([])
    
    #for i in range(100, 550, 50):
    for i in range(100,510,10):
        for j in range(1,NUM_FILES+1):
            freq_path = os.path.join(freq_dir, str(i), split, str(j) + ".csv")
            time_series = pd.read_csv(freq_path, header=None).to_numpy()
            time_series = time_series[:,:NUM_SAMPLES]
           
                
            NUM_CHANNELS = time_series.shape[0]
            
            if not first:
                final_time_series = np.concatenate((final_time_series, time_series))
                final_time_labels = np.concatenate((final_time_labels, [i]*NUM_CHANNELS))
            else:
                final_time_series = time_series
                final_time_labels = np.array([i]*NUM_CHANNELS)
                first = False

    return final_time_series, final_time_labels
"""

"""
def get_data(split):

    NUM_SAMPLES = 80 #70
    NUM_FRAMES = 1

    first = True
    
    #pdb.set_trace()
    freq_dir = "../data3"
    NUM_FILES = len(os.listdir(freq_dir + '/' + str(100) + '/' + split))
    EFFECTIVE_FILES = NUM_FILES - NUM_FILES % NUM_FRAMES

    final_time_series = np.array([])
    final_time_labels = np.array([])
    
    #for i in range(100, 550, 50):
    for i in range(100,510,10):
        for j in range(1,NUM_FILES+1):
            freq_path = os.path.join(freq_dir, str(i), split, str(j) + ".csv")
            time_series = pd.read_csv(freq_path, header=None).to_numpy()
            time_series = np.expand_dims(time_series[0][:NUM_SAMPLES],0)
            
            \"""
            if not first:
                final_time_series = np.concatenate((final_time_series, time_series))
                final_time_labels = np.concatenate((final_time_labels, [i]))
            else:
                final_time_series = time_series
                final_time_labels = np.array([i])
                first = False
            \"""
                
            if NUM_FRAMES > 1:
                if not first:

                    if j % NUM_FRAMES == 1:
                        tmp_time_series = time_series
                        final_time_labels = np.concatenate((final_time_labels, [i]))
                    else:
                        tmp_time_series = np.concatenate((tmp_time_series, time_series),1)
                    if j % NUM_FRAMES == 0:
                        final_time_series = np.concatenate((final_time_series, np.expand_dims(tmp_time_series, 0)))
                else:

                    if j % NUM_FRAMES == 1:
                        tmp_time_series = time_series
                        final_time_labels = np.array([i])
                    else:
                        tmp_time_series = np.concatenate((tmp_time_series, time_series),1)
                    if j % NUM_FRAMES == 0:
                        final_time_series = np.expand_dims(tmp_time_series, 0)
                        first = False
                if j == EFFECTIVE_FILES:
                    break
            else:
                if not first:
                    final_time_series = np.concatenate((final_time_series, [time_series]))
                    final_time_labels = np.concatenate((final_time_labels, [i]))
                else:
                    final_time_series = np.expand_dims(time_series,0)
                    final_time_labels = np.array([i])
                    first = False


    return final_time_series, final_time_labels
"""
def load_dataset_at(index, NUM_FRAMES, MAX_SEQUENCE_LENGTH, normalize_timeseries=False, verbose=True, evaluation=False) -> (np.array, np.array):
    """
    Loads a Univaraite UCR Dataset indexed by `utils.constants`.

    Args:
        index: Integer index, set inside `utils.constants` that refers to the
            dataset.
        normalize_timeseries: Bool / Integer. Determines whether to normalize
            the timeseries.

            If False, does not normalize the time series.
            If True / int not equal to 2, performs standard sample-wise
                z-normalization.
            If 2: Performs full dataset z-normalization.
        verbose: Whether to describe the dataset being loaded.

    Returns:
        A tuple of shape (X_train, y_train, X_test, y_test, is_timeseries).
        For legacy reasons, is_timeseries is always True.
    """
    """
    assert index < len(TRAIN_FILES), "Index invalid. Could not load dataset at %d" % index
    if verbose: print("Loading train / test dataset : ", TRAIN_FILES[index], TEST_FILES[index])
    """
    
    #pdb.set_trace()
    if evaluation:
        split = "test" #"validation" #"test"
        X_train = []
        y_train = []
    
    else:
        split = "train"
        
        X_train, y_train = get_data(split, NUM_FRAMES, MAX_SEQUENCE_LENGTH)
        
        split = "validation"
    
    X_test, y_test = get_data(split, NUM_FRAMES, MAX_SEQUENCE_LENGTH)


    is_timeseries = True # assume all input data is univariate time series


    # extract labels Y and normalize to [0 - (MAX - 1)] range

    #y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min())

    # drop labels column from train set X
    #df.drop(df.columns[0], axis=1, inplace=True)



    #X_train = X_train[:, np.newaxis, :]
    # scale the values
    #pdb.set_trace()
    if normalize_timeseries:
        normalize_timeseries = int(normalize_timeseries)

        if normalize_timeseries == 2:
            X_train_mean = X_train.mean()
            X_train_std = X_train.std()
            X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)

        else:
            X_train_mean = X_train.mean(axis=-1, keepdims=True)
            X_train_std = X_train.std(axis=-1, keepdims=True)
            X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)

    if verbose: print("Finished loading train dataset..")



    # remove all columns which are completely empty




    # extract labels Y and normalize to [0 - (MAX - 1)] range


    #y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min())

    # drop labels column from train set X





    #X_test = X_test[:, np.newaxis, :]
    # scale the values
    if normalize_timeseries:
        normalize_timeseries = int(normalize_timeseries)

        if normalize_timeseries == 2:
            X_test = (X_test - X_train_mean) / (X_train_std + 1e-8)
        else:
            X_test_mean = X_test.mean(axis=-1, keepdims=True)
            X_test_std = X_test.std(axis=-1, keepdims=True)
            X_test = (X_test - X_test_mean) / (X_test_std + 1e-8)

    if verbose and not evaluation:
        print("Finished loading test dataset..")
        print()
        print("Number of train samples : ", X_train.shape[0], "Number of test samples : ", X_test.shape[0])

        print("Sequence length : ", X_train.shape[-1])

    return X_train, y_train, X_test, y_test, is_timeseries


def calculate_dataset_metrics(X_train):
    """
    Calculates the dataset metrics used for model building and evaluation.

    Args:
        X_train: The training dataset.

    Returns:
        A tuple of (None, sequence_length). None is for legacy
        purposes.
    """
    is_timeseries = len(X_train.shape) == 3
    if is_timeseries:
        # timeseries dataset
        max_sequence_length = X_train.shape[-1]
        max_nb_words = None
    else:
        # transformed dataset
        max_sequence_length = X_train.shape[-1]
        max_nb_words = np.amax(X_train) + 1

    return max_nb_words, max_sequence_length


def plot_dataset(dataset_id, seed=None, limit=None, cutoff=None,
                 normalize_timeseries=False, plot_data=None,
                 type='Context', plot_classwise=False):
    """
    Util method to plot a dataset under several possibilities.

    Args:
        dataset_id: Integer id, refering to the dataset set inside
            `utils/constants.py`.
        seed: Numpy Random seed.
        limit: Number of data points to be visualized. Min of 1.
        cutoff: Optional integer which slices of the first `cutoff` timesteps
            from the input signal.
        normalize_timeseries: Bool / Integer. Determines whether to normalize
            the timeseries.

            If False, does not normalize the time series.
            If True / int not equal to 2, performs standard sample-wise
                z-normalization.
            If 2: Performs full dataset z-normalization.
        plot_data: Additional data used for plotting in place of the
            loaded train set. Can be the test set or some other val set.
        type: Type of plot being built. Can be one of ['Context', any other string].
            Context is a specific keyword, used for Context from Attention LSTM.
            If any other string is provided, it is used in the title.
        plot_classwise: Bool flag. Wheter to visualize the samples
            seperated by class. When doing so, `limit` is multiplied by
            the number of classes so it is better to set `limit` to 1 in
            such cases
    """
    np.random.seed(seed)

    if plot_data is None:
        X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(
                                                               dataset_id,
                                                               normalize_timeseries=normalize_timeseries)

        if not is_timeseries:
            print("Can plot time series input data only!\n"
                  "Continuing without plot!")
            return

        max_nb_words, sequence_length = calculate_dataset_metrics(X_train)

        if sequence_length != MAX_SEQUENCE_LENGTH_LIST[dataset_id]:
            if cutoff is None:
                choice = cutoff_choice(dataset_id, sequence_length)
            else:
                assert cutoff in ['pre', 'post'], 'Cutoff parameter value must be either "pre" or "post"'
                choice = cutoff

            if choice not in ['pre', 'post']:
                return
            else:
                X_train, X_test = X_test(X_train, X_test, choice, dataset_id, sequence_length)

        X_train_attention = None
        X_test_attention = None

    else:
        X_train, y_train, X_test, y_test, X_train_attention, X_test_attention = plot_data

    if limit is None:
        train_size = X_train.shape[0]
        test_size = X_test.shape[0]
    else:
        if not plot_classwise:
            train_size = limit
            test_size = limit
        else:
            assert limit == 1, 'If plotting classwise, limit must be 1 so as to ensure number of samples per class = 1'
            train_size = NB_CLASSES_LIST[dataset_id] * limit
            test_size = NB_CLASSES_LIST[dataset_id] * limit

    if not plot_classwise:
        train_idx = np.random.randint(0, X_train.shape[0], size=train_size)
        X_train = X_train[train_idx, 0, :]
        X_train = X_train.transpose((1, 0))

        if X_train_attention is not None:
            X_train_attention = X_train_attention[train_idx, 0, :]
            X_train_attention = X_train_attention.transpose((1, 0))
    else:
        classwise_train_list = []
        for y_ in sorted(np.unique(y_train[:, 0])):
            class_train_idx = np.where(y_train[:, 0] == y_)
            classwise_train_list.append(class_train_idx[:])

        classwise_sample_size_list = [len(x[0]) for x in classwise_train_list]
        size = min(classwise_sample_size_list)
        train_size = min([train_size // NB_CLASSES_LIST[dataset_id], size])

        for i in range(len(classwise_train_list)):
            classwise_train_idx = np.random.randint(0, len(classwise_train_list[i][0]), size=train_size)
            classwise_train_list[i] = classwise_train_list[i][0][classwise_train_idx]

        classwise_X_train_list = []
        classwise_X_train_attention_list = []

        for classwise_train_idx in classwise_train_list:
            classwise_X = X_train[classwise_train_idx, 0, :]
            classwise_X = classwise_X.transpose((1, 0))
            classwise_X_train_list.append(classwise_X)

            if X_train_attention is not None:
                classwise_X_attn = X_train_attention[classwise_train_idx, 0, :]
                classwise_X_attn = classwise_X_attn.transpose((1, 0))
                classwise_X_train_attention_list.append(classwise_X_attn)

        classwise_X_train_list = [np.asarray(x) for x in classwise_X_train_list]
        classwise_X_train_attention_list = [np.asarray(x) for x in classwise_X_train_attention_list]

        # classwise x train
        X_train = np.concatenate(classwise_X_train_list, axis=-1)

        # classwise x train attention
        if X_train_attention is not None:
            X_train_attention = np.concatenate(classwise_X_train_attention_list, axis=-1)

    if not plot_classwise:
        test_idx = np.random.randint(0, X_test.shape[0], size=test_size)
        X_test = X_test[test_idx, 0, :]
        X_test = X_test.transpose((1, 0))

        if X_test_attention is not None:
            X_test_attention = X_test_attention[test_idx, 0, :]
            X_test_attention = X_test_attention.transpose((1, 0))
    else:
        classwise_test_list = []
        for y_ in sorted(np.unique(y_test[:, 0])):
            class_test_idx = np.where(y_test[:, 0] == y_)
            classwise_test_list.append(class_test_idx[:])

        classwise_sample_size_list = [len(x[0]) for x in classwise_test_list]
        size = min(classwise_sample_size_list)
        test_size = min([test_size // NB_CLASSES_LIST[dataset_id], size])

        for i in range(len(classwise_test_list)):
            classwise_test_idx = np.random.randint(0, len(classwise_test_list[i][0]), size=test_size)
            classwise_test_list[i] = classwise_test_list[i][0][classwise_test_idx]

        classwise_X_test_list = []
        classwise_X_test_attention_list = []

        for classwise_test_idx in classwise_test_list:
            classwise_X = X_test[classwise_test_idx, 0, :]
            classwise_X = classwise_X.transpose((1, 0))
            classwise_X_test_list.append(classwise_X)

            if X_test_attention is not None:
                classwise_X_attn = X_test_attention[classwise_test_idx, 0, :]
                classwise_X_attn = classwise_X_attn.transpose((1, 0))
                classwise_X_test_attention_list.append(classwise_X_attn)

        classwise_X_test_list = [np.asarray(x) for x in classwise_X_test_list]
        classwise_X_test_attention_list = [np.asarray(x) for x in classwise_X_test_attention_list]

        # classwise x test
        X_test = np.concatenate(classwise_X_test_list, axis=-1)

        # classwise x test attention
        if X_test_attention is not None:
            X_test_attention = np.concatenate(classwise_X_test_attention_list, axis=-1)

    print('X_train shape : ', X_train.shape)
    print('X_test shape : ', X_test.shape)

    columns = ['Class %d' % (i + 1) for i in range(X_train.shape[1])]
    train_df = pd.DataFrame(X_train,
                            index=range(X_train.shape[0]),
                            columns=columns)

    test_df = pd.DataFrame(X_test,
                           index=range(X_test.shape[0]),
                           columns=columns)

    if plot_data is not None:
        rows = 2
        cols = 2
    else:
        rows = 1
        cols = 2

    fig, axs = plt.subplots(rows, cols, squeeze=False,
                           tight_layout=True, figsize=(8, 6))
    axs[0][0].set_title('Train dataset', size=16)
    axs[0][0].set_xlabel('timestep')
    axs[0][0].set_ylabel('value')
    train_df.plot(subplots=False,
                  legend='best',
                  ax=axs[0][0],)

    axs[0][1].set_title('Test dataset', size=16)
    axs[0][1].set_xlabel('timestep')
    axs[0][1].set_ylabel('value')
    test_df.plot(subplots=False,
                 legend='best',
                 ax=axs[0][1],)

    if plot_data is not None and X_train_attention is not None:
        columns = ['Class %d' % (i + 1) for i in range(X_train_attention.shape[1])]
        train_attention_df = pd.DataFrame(X_train_attention,
                            index=range(X_train_attention.shape[0]),
                            columns=columns)

        axs[1][0].set_title('Train %s Sequence' % (type), size=16)
        axs[1][0].set_xlabel('timestep')
        axs[1][0].set_ylabel('value')
        train_attention_df.plot(subplots=False,
                                legend='best',
                                ax=axs[1][0])

    if plot_data is not None and X_test_attention is not None:
        columns = ['Class %d' % (i + 1) for i in range(X_test_attention.shape[1])]
        test_df = pd.DataFrame(X_test_attention,
                               index=range(X_test_attention.shape[0]),
                               columns=columns)

        axs[1][1].set_title('Test %s Sequence' % (type), size=16)
        axs[1][1].set_xlabel('timestep')
        axs[1][1].set_ylabel('value')
        test_df.plot(subplots=False,
                     legend='best',
                     ax=axs[1][1])

    plt.show()


def cutoff_choice(dataset_id, sequence_length):
    """
    Helper to allow the user to select whether they want to cutoff timesteps or not,
    and in what manner (pre or post).

    Args:
        dataset_id: Dataset ID
        sequence_length: Length of the sequence originally.

    Returns:
        String choice of pre or post slicing.
    """
    print("Original sequence length was :", sequence_length, "New sequence Length will be : ",
          MAX_SEQUENCE_LENGTH_LIST[dataset_id])
    choice = input('Options : \n'
                   '`pre` - cut the sequence from the beginning\n'
                   '`post`- cut the sequence from the end\n'
                   '`anything else` - stop execution\n'
                   'To automate choice: add flag `cutoff` = choice as above\n'
                   'Choice = ')

    choice = str(choice).lower()
    return choice


def cutoff_sequence(X_train, X_test, choice, dataset_id, sequence_length):
    """
    Slices of the first `cutoff` timesteps from the input signal.

    Args:
        X_train: Train sequences.
        X_test: Test sequences.
        choice: User's choice of slicing method.
        dataset_id: Integer id of the dataset set inside `utils/constants.py`.
        sequence_length: Original length of the sequence.

    Returns:
        A tuple of (X_train, X_test) after slicing off the requisit number of
        timesteps.
    """
    assert MAX_SEQUENCE_LENGTH_LIST[dataset_id] < sequence_length, "If sequence is to be cut, max sequence" \
                                                                   "length must be less than original sequence length."
    cutoff = sequence_length - MAX_SEQUENCE_LENGTH_LIST[dataset_id]
    if choice == 'pre':
        if X_train is not None:
            X_train = X_train[:, :, cutoff:]
        if X_test is not None:
            X_test = X_test[:, :, cutoff:]
    else:
        if X_train is not None:
            X_train = X_train[:, :, :-cutoff]
        if X_test is not None:
            X_test = X_test[:, :, :-cutoff]
    print("New sequence length :", MAX_SEQUENCE_LENGTH_LIST[dataset_id])
    return X_train, X_test


if __name__ == "__main__":
    word_list = []
    seq_len_list = []
    classes = []

    # for index in range(85, 128):
    #
    #     x, y, x_test, y_test, is_timeseries = load_dataset_at(index)
    #     nb_words, seq_len = calculate_dataset_metrics(x)
    #     print("-" * 80)
    #     print("Dataset : ", index + 1)
    #     print("Train :: X shape : ", x.shape, "Y shape : ", y.shape, "Nb classes : ", len(np.unique(y)))
    #     print("Test :: X shape : ", x_test.shape, "Y shape : ", y_test.shape, "Nb classes : ", len(np.unique(y)))
    #     print("Classes : ", np.unique(y))
    #     print()
    #
    #     word_list.append(nb_words)
    #     seq_len_list.append(seq_len)
    #     classes.append(len(np.unique(y)))
    #
    # print("Word List : ", word_list)
    # print("Sequence length list : ", seq_len_list)
    # print("Max number of classes : ", classes)

    plot_dataset(dataset_id=77, seed=1, limit=1, cutoff=None, normalize_timeseries=True,
                 plot_classwise=True)
