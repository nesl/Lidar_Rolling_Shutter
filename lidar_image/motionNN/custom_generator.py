import tensorflow as tf
import numpy as np

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df, frames,
                 batch_size,
                 input_size=(224, 224, 3),
                 rescale=1./65535,
                 shuffle=True,
                 augment=False):
        
        self.df = df.copy()
        self.frames = frames
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.rescale = rescale
        self.labels = self.df["class"].to_numpy()
        self.augment = augment
        
        self.n = len(self.df)
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __get_input(self, paths):
    
        #xmin, ymin, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        images_arr = np.array([])
        for f in range(self.frames):
            image = tf.keras.preprocessing.image.load_img(paths[f], grayscale=True)
            image_arr = tf.keras.preprocessing.image.img_to_array(image)*self.rescale
            if f == 0:
                images_arr = image_arr
            else:
                images_arr =  np.concatenate((images_arr, image_arr), axis=2)
            
        #image_arr = image_arr[ymin:ymin+h, xmin:xmin+w]
        #image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()
        if self.augment:
            images_arr = tf.keras.preprocessing.image.random_shift(images_arr, 0.05, 0.05, row_axis=0, col_axis=1, channel_axis=2)
            images_arr = tf.keras.preprocessing.image.random_zoom(images_arr, (0.75,1.25), row_axis=0,col_axis=1, channel_axis=2)

        return images_arr
    
    def __get_output(self, label, num_classes):
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)
    
    def __get_data(self, batches):
        # Generates data containing batch_size samples

        X_batch = []
        Y_batch = []
        for _,b in batches.iterrows():
            files = [str(b['prefix']) + '/' + b['filename'+str(f)] for f in range(self.frames)]
            X_batch.append(self.__get_input(files))
            Y_batch.append(float(b['class']))


        X_batch = np.asarray(X_batch)
        Y_batch = np.asarray(Y_batch)

        return X_batch, Y_batch
    
    def __getitem__(self, index):
        
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size

