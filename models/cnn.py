from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

class ConvNet:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=self.config.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.config.num_classes))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy',
            optimizer=self.config.optimizer,
            metrics=['accuracy'])


    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def get_model(self):
        return self.model