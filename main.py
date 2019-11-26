from models.cnn import ConvNet
from trainer.trainer import ModelTrainer
from data_loader.load_cifar import CifarDataLoader
from config.config import Config
import os

SAVE_DIR = os.path.join(os.getcwd(),'saved_models')

config_default = {
    'batch_size' : 32,
    'num_classes' : 10,
    'epochs' : 50,
    'input_shape' : (32, 32, 3),
    'optimizer' : 'not_set' # will be set later
}

optimizers = ['adam', 'adagrad', 'sgd']

def main():

    # Loop over multiple optimizers
    for optimizer in optimizers:

        # Select optimizer
        config_default['optimizer'] = optimizer

        # Load config
        config = Config(config_default)

        # Load data
        load_data = CifarDataLoader(config)
        train_data = load_data.get_train_data()
        validation_data = load_data.get_test_data()

        # Create Model
        model = ConvNet(config).get_model() # without dropout
        model_do = ConvNetDropout(config).get_model() # with dropout

        # Train Model
        trainer = ModelTrainer(model, train_data, validation_data, config) # without dropout
        trainer.train()

        trainer_do = ModelTrainer(model_do, train_data, validation_data, config) # with dropout
        trainer_do.train()

        # Save trained model
        model_name = 'cnn_' + optimizer + '.h5' # without dropout
        save_model = os.path.join(SAVE_DIR,model_name)
        trainer.save(save_model)

        model_do_name = 'cnn_dropout' + optimizer + '.h5' # with dropout
        save_model_do = os.path.join(SAVE_DIR,model_do_name)
        trainer_do.save(save_model_do)

        # Print the results

        print("optimizer: ", optimizer)

        print("Without dropout")
        print("loss: ", trainer.loss)
        print("accuracy: ", trainer.acc)
        print("validation loss: ", trainer.val_loss)
        print("validation accuracy: ", trainer.val_acc)

        print("With dropout")
        print("loss: ", trainer_do.loss)
        print("accuracy: ", trainer_do.acc)
        print("validation loss: ", trainer_do.val_loss)
        print("validation accuracy: ", trainer_do.val_acc)

if __name__ == '__main__':
    main()