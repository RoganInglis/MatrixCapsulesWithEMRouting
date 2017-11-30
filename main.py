import os
import json
import time
import tensorflow as tf
import numpy as np
import models

# See the __init__ script in the models folder
# `make_model` is a helper function to load any models you have
from models import make_model

# I personally always like to make my paths absolute
# to be independent from where the python binary is called
project_dir = os.path.dirname(os.path.realpath(__file__))

flags = tf.app.flags


# Hyper-parameters search configuration
flags.DEFINE_boolean('fullsearch', False, 'Perform a full search of hyperparameter space ex:(hyperband -> lr search -> hyperband with best lr)')
flags.DEFINE_boolean('dry_run', False, 'Perform a dry_run (testing purpose)')
flags.DEFINE_integer('nb_process', 4, 'Number of parallel process to perform a HP search')

# fixed_params is a trick I use to be able to fix some parameters inside the model random function
# For example, one might want to explore different models fixing the learning rate,
# see the basic_model get_random_config function
flags.DEFINE_string('fixed_params', "{}", 'JSON inputs to fix some params in a HP search, ex: \'{"lr": 0.001}\'')

# Model configuration
flags.DEFINE_string('model_name', 'BasicModel', 'Unique name of the model')
flags.DEFINE_boolean('best', False, 'Force to use the best known configuration')
flags.DEFINE_float('learning_rate', 1e-3, 'The learning rate of SGD')
flags.DEFINE_float('drop_keep_prob', 1.0, 'The dropout keep probability')
flags.DEFINE_float('l2', 0.0, 'L2 regularisation strength')
flags.DEFINE_integer('batch_size', 64, 'Batch size')

# Training configuration
flags.DEFINE_boolean('infer', False, 'Load model for inference')
flags.DEFINE_boolean('debug', False, 'Debug mode')
flags.DEFINE_integer('max_iter', 1000000, 'Max number of training iterations')
flags.DEFINE_integer('max_train_epochs', 1000, 'Max number of training epochs')
flags.DEFINE_boolean('test', False, 'Load a model and compute test performance')
flags.DEFINE_integer('test_every', 10, 'Epoch interval at which to test the model during training')

# This is very important for TensorBoard
# each model will end up in its own unique folder using time module
# Obviously one can also choose to name the output folder
flags.DEFINE_string('result_dir', project_dir + '/results/' + flags.FLAGS.model_name + '/' +
                    str(int(time.time())),
                    'Name of the directory to store/log the model (if it exists, the model will be loaded from it)')
flags.DEFINE_string('test_result_dir', project_dir + '/results/' + flags.FLAGS.model_name + '/' +
                    str(int(time.time())) + '/test',
                    'Name of the directory to store/log the model test results (for TensorBoard)')

# Another important point, you must provide an access to the random seed
# to be able to fully reproduce an experiment
flags.DEFINE_integer('random_seed', np.random.randint(0, 2**8), 'Value of random seed')


def main(_):
    config = flags.FLAGS.__flags.copy()
    # fixed_params must be a string to be passed in the shell, let's use JSON
    config["fixed_params"] = json.loads(config["fixed_params"])

    if config['fullsearch']:
        print('Hyperparameter search not implemented yet')
    else:
        model = make_model(config)

        if config['infer']:
            # Some code for inference ...
            model.infer()
        elif config['test']:
            model.test()
        else:
            # Some code for training ...
            model.train()


if __name__ == '__main__':
    tf.app.run()

