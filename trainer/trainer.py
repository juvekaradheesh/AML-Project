class ModelTrainer:
    def __init__(self, model, train_data, validation_data, config):
        self.model = model
        self.train_data = train_data
        self.validation_data = validation_data
        self.config = config
        self.loss = []
        self.val_loss = []

    def train(self):
        history = self.model.fit(
            self.train_data[0], self.train_data[1],
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_data=self.validation_data,
            shuffle=True)

        self.loss.extend(history.history['loss'])
        self.val_loss.extend(history.history['val_loss'])
    
    def get_trained_model(self):
        return self.model
    
    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")