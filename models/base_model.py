import os
import copy
import json
import tensorflow as tf
import numpy as np
from models import utils
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug


class BaseModel(object):
    # To build your agent, you only to pass a "configuration" which is a dictionary
    def __init__(self, config):
        # I like to keep the best HP found so far inside the model itself
        # This is a mechanism to load the best HP and override the configuration
        if config['best']:
            config.update(self.get_best_config())

        # I make a `deepcopy` of the configuration before using it
        # to avoid any potential mutation when I iterate asynchronously over configurations
        self.config = copy.deepcopy(config)

        if config['debug']:  # This is a personal check i like to do
            print('config', self.config)

        # When working with NN, one usually initialize randomly
        # and you want to be able to reproduce your initialization so make sure
        # you store the random seed and actually use it in your TF graph (tf.set_random_seed() for example)
        self.random_seed = self.config['random_seed']

        # All models share some basics hyper parameters, this is the section where we
        # copy them into the model
        self.model_name = self.config['model_name']
        self.result_dir = self.config['result_dir']
        self.validation_result_dir = self.config['validation_result_dir']
        self.max_iter = self.config['max_iter']
        self.max_train_epochs = self.config['max_train_epochs']
        self.drop_keep_prob = self.config['drop_keep_prob']
        self.learning_rate = self.config['learning_rate']
        self.l2 = self.config['l2']
        self.batch_size = self.config['batch_size']
        self.image_dim = self.config['image_dim']
        self.save_every = self.config['save_every']
        self.test_every = self.config['test_every']
        self.train_summary_every = self.config['train_summary_every']
        self.validation_summary_every = self.config['validation_summary_every']
        self.n_classes = self.config['n_classes']

        self.debug = self.config['debug']

        # Load data
        self.data = input_data.read_data_sets('MNIST_data', one_hot=True)

        # Now the child Model needs some custom parameters, to avoid any
        # inheritance hell with the __init__ function, the model
        # will override this function completely
        self.set_model_props(config)

        # Set up global step
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # Again, child Model should provide its own build_graph function
        self.graph = self.build_graph(self.graph)

        # Any operations that should be in the graph but are common to all models
        # can be added this way, here
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep=50)
            self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Add all the other common code for the initialization here
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=sess_config, graph=self.graph)
        self.train_summary_writer = tf.summary.FileWriter(self.result_dir, self.sess.graph)
        self.validation_summary_writer = tf.summary.FileWriter(self.validation_result_dir, self.sess.graph)

        # This function is not always common to all models, that's why it's again
        # separated from the __init__ one
        self.init()

    def set_model_props(self, config):
        # This function is here to be overridden completely.
        # When you look at your model, you want to know exactly which custom options it needs.
        pass

    def get_best_config(self):
        # This function is here to be overridden completely.
        # It returns a dictionary used to update the initial configuration (see __init__)
        return {}

    @staticmethod
    def get_random_config(fixed_params={}):
        # Why static? Because you want to be able to pass this function to other processes
        # so they can independently generate random configuration of the current model
        raise Exception('The get_random_config function must be overriden by the agent')

    def build_graph(self, graph):
        raise Exception('The build_graph function must be overriden by the agent')

    def infer(self, audio_input):
        raise Exception('The infer function must be overriden by the agent')

    def test(self):
        raise Exception('The test function must be overriden by the agent')

    def learn_from_epoch(self):
        # I like to separate the function to train per epoch and the function to train globally
        raise Exception('The learn_from_epoch function must be overriden by the agent')

    def train(self):
        # This function is usually common to all your models
        for self.epoch_id in range(0, self.max_train_epochs):
            # Perform all TensorBoard operations within learn_from_episode
            self.learn_from_epoch()

            # If you don't want to save during training, you can just pass a negative number
            if self.save_every > 0 and self.epoch_id % self.save_every == 0:
                self.save()

    def save(self):
        # This function is usually common to all your models, Here is an example:
        if self.config['debug']:
            print('Saving to %s' % self.result_dir)
        self.saver.save(self.sess, self.result_dir + '/model-ep_' + str(self.epoch_id))

        # I always keep the configuration that
        if not os.path.isfile(self.result_dir + '/config.json'):
            config = self.config
            if 'phi' in config:
                del config['phi']
            with open(self.result_dir + '/config.json', 'w') as f:
                json.dump(self.config, f)

    def init(self):
        # This function is usually common to all your models
        # but making separate than the __init__ function allows it to be overidden cleanly
        # this is an example of such a function
        checkpoint = tf.train.get_checkpoint_state(self.result_dir)
        tf.train.write_graph(self.graph, self.result_dir, self.model_name + ".pbtxt")
        if checkpoint is None:
            self.sess.run(self.init_op)
        else:
            if self.config['debug']:
                print('Loading the model from folder: %s' % self.result_dir)
            self.sess.run(self.init_op)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
        if self.debug:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            self.sess.add_tensor_filter('has_inf_or_nan', tf_debug.has_inf_or_nan)
