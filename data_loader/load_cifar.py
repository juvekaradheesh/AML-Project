from keras.datasets import cifar10
import keras.utils

class CifarDataLoader:
    def __init__(self, config):
        self.config = config
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.y_train = keras.utils.to_categorical(self.y_train, self.config.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.config.num_classes)
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255

    def get_train_data(self):
        return self.x_train, self.y_train

    def get_test_data(self):
        return self.x_test, self.y_test