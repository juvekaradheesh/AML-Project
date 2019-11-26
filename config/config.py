class Config:

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.num_classes = config['num_classes']
        self.epochs = config['epochs']
        self.input_shape = config['input_shape']
        self.optimizer = config['optimizer']