import numpy as np
import os
import random
import tensorflow as tf
from keras import backend as K

from models.cnn import ConvNet
from models.cnn_dropout import ConvNetDropout
from trainer.trainer import ModelTrainer
from data_loader.load_cifar import CifarDataLoader
from config.config import Config


SAVE_DIR = os.path.join(os.getcwd(),'saved_models')

config_default = {
    'batch_size' : 32,
    'num_classes' : 10,
    'epochs' : 75,
    'input_shape' : (32, 32, 3),
    'optimizer' : 'not_set' # will be set later
}

RESULTS_FILE = "results.txt"
SEED = 0

def main():

    f = open(RESULTS_FILE, "a")
    
    # Load config
    config = Config(config_default)

    # Load data
    load_data = CifarDataLoader(config)
    train_data = load_data.get_train_data()
    validation_data = load_data.get_test_data()

    optimizers = ['adam', 'adagrad', 'sgd']

    # Loop over multiple optimizers
    # Without dropout
    for optimizer in optimizers:

        # Set seed value
        os.environ['PYTHONHASHSEED']=str(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        tf.compat.v1.set_random_seed(SEED)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        K.set_session(sess)

        # Set optimizer
        config_default['optimizer'] = optimizer

        # Load config
        config = Config(config_default)

        # Create model
        temp = ConvNet(config)
        model = temp.get_model() # without dropout

        # Train model
        trainer = ModelTrainer(model, train_data, validation_data, config) # without dropout
        trainer.train()

        # Save trained model
        model_name = 'cnn_' + optimizer + '.h5' # without dropout
        save_model = os.path.join(SAVE_DIR,model_name)
        trainer.save(save_model)

        # Print the results
        print("optimizer: ", optimizer)
        print("Without dropout")
        print("loss: ", trainer.loss)
        print("validation loss: ", trainer.val_loss)

        f.write("optimizer: " + optimizer + "\n\n")
        f.write("Without dropout \n")
        f.write("loss: " + str(trainer.loss) + "\n")
        f.write("validation loss: " + str(trainer.val_loss) + "\n")
        f.write("\n")

    # Loop over multiple optimizers
    # With dropout
    for optimizer in optimizers:

        # Set seed value
        os.environ['PYTHONHASHSEED']=str(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        tf.set_random_seed(SEED)
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
        
        # Set optimizer
        config_default['optimizer'] = optimizer

        # Load config
        config = Config(config_default)

        # Create model
        temp = ConvNetDropout(config)
        model_do = temp.get_model() # with dropout

        # Train model
        trainer_do = ModelTrainer(model_do, train_data, validation_data, config) # with dropout
        trainer_do.train()

        # Save trained model
        model_do_name = 'cnn_dropout' + optimizer + '.h5' # with dropout
        save_model_do = os.path.join(SAVE_DIR,model_do_name)
        trainer_do.save(save_model_do)

        # Print the results
        print("optimizer: ", optimizer)
        print("With dropout")
        print("loss: ", trainer_do.loss)
        print("validation loss: ", trainer_do.val_loss)

        f.write("optimizer: " + optimizer + "\n\n")
        f.write("With dropout \n")
        f.write("loss: " + str(trainer_do.loss) + "\n")
        f.write("validation loss: " + str(trainer_do.val_loss) + "\n")
        f.write("\n")



        
    f.close()
if __name__ == '__main__':
    main()