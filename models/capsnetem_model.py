from models import BaseModel
from models import utils
import numpy as np
from tqdm import trange
import tensorflow as tf


class CapsNetEMModel(BaseModel):
    def set_model_props(self, config):
        # ReLU Conv1
        self.relu_conv1_kernel_size = self.config['relu_conv1_kernel_size']
        self.relu_conv1_filters = self.config['relu_conv1_filters']
        self.relu_conv1_stride = self.config['relu_conv1_stride']

        # PrimaryCaps
        self.primarycaps_out_capsules = self.config['primarycaps_out_capsules']
        self.pose_size = self.config['pose_size']

        # ConvCaps1
        self.convcaps1_out_capsules = self.config['convcaps1_out_capsules']
        self.convcaps1_kernel_size = self.config['convcaps1_kernel_size']
        self.convcaps1_strides = self.config['convcaps1_strides']
        self.convcaps1_padding = self.config['convcaps1_padding']
        self.convcaps1_n_routing_iterations = self.config['convcaps1_n_routing_iterations']
        self.convcaps1_init_inverse_temp = self.config['convcaps1_init_inverse_temp']
        self.convcaps1_final_inverse_temp = self.config['convcaps1_final_inverse_temp']

        # ConvCaps2
        self.convcaps2_out_capsules = self.config['convcaps2_out_capsules']
        self.convcaps2_kernel_size = self.config['convcaps2_kernel_size']
        self.convcaps2_strides = self.config['convcaps2_strides']
        self.convcaps2_padding = self.config['convcaps2_padding']
        self.convcaps2_n_routing_iterations = self.config['convcaps2_n_routing_iterations']
        self.convcaps2_init_inverse_temp = self.config['convcaps2_init_inverse_temp']
        self.convcaps2_final_inverse_temp = self.config['convcaps2_final_inverse_temp']

        # Class Capsules
        self.classcaps_n_routing_iterations = self.config['classcaps_n_routing_iterations']
        self.classcaps_init_inverse_temp = self.config['classcaps_init_inverse_temp']
        self.classcaps_final_inverse_temp = self.config['classcaps_final_inverse_temp']

        # Spread Loss
        self.initial_margin = self.config['initial_margin']
        self.final_margin = self.config['final_margin']
        self.margin_decay_steps = self.config['margin_decay_steps']

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
        with graph.as_default():
            # Create placeholders
            self.placeholders = {'image': tf.placeholder(tf.float32, [None, self.image_dim], name='image'),
                                 'label': tf.placeholder(tf.int32, [None, self.n_classes], name='label')}

            # Define main model graph
            self.margin = tf.train.polynomial_decay(self.initial_margin, self.global_step, self.margin_decay_steps,
                                                    self.final_margin)

            # Set up parameter dicts
            relu_conv1_params = {'kernel_size': self.relu_conv1_kernel_size,
                                 'filters': self.relu_conv1_filters,
                                 'strides': self.relu_conv1_stride}

            primarycaps_params = {'out_capsules': self.primarycaps_out_capsules,
                                  'pose_size': self.pose_size}

            convcaps1_params = {'out_capsules': self.convcaps1_out_capsules,
                                'kernel_size': self.convcaps1_kernel_size,
                                'strides': self.convcaps1_strides,
                                'padding': self.convcaps1_padding,
                                'n_routing_iterations': self.convcaps1_n_routing_iterations,
                                'init_inverse_temp': self.convcaps1_init_inverse_temp,
                                'final_inverse_temp': self.convcaps1_final_inverse_temp}

            convcaps2_params = {'out_capsules': self.convcaps2_out_capsules,
                                'kernel_size': self.convcaps2_kernel_size,
                                'strides': self.convcaps2_strides,
                                'padding': self.convcaps2_padding,
                                'n_routing_iterations': self.convcaps2_n_routing_iterations,
                                'init_inverse_temp': self.convcaps2_init_inverse_temp,
                                'final_inverse_temp': self.convcaps2_final_inverse_temp}

            classcaps_params = {'n_classes': self.n_classes,
                                'n_routing_iterations': self.classcaps_n_routing_iterations,
                                'init_inverse_temp': self.classcaps_init_inverse_temp,
                                'final_inverse_temp': self.classcaps_final_inverse_temp}

            spread_loss_params = {'margin': self.margin}

            self.loss, self.predictions, self.accuracy, self.correct, self.summaries = utils.build_capsnetem_graph(
                                                                                                   self.placeholders,
                                                                                                   relu_conv1_params,
                                                                                                   primarycaps_params,
                                                                                                   convcaps1_params,
                                                                                                   convcaps2_params,
                                                                                                   classcaps_params,
                                                                                                   spread_loss_params,
                                                                                                   image_dim=self.image_dim)

            # Define optimiser
            self.optim = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optim.minimize(self.loss, global_step=self.global_step)

            # Set up summaries
            """
            self.train_summary = tf.summary.merge([self.summaries['accuracy'],
                                                   self.summaries['loss'],
                                                   *self.summaries['general']])
            """
            self.train_summary = tf.summary.merge_all()
            self.validation_summary = tf.summary.merge([self.summaries['accuracy'],
                                                       self.summaries['loss']])

        return graph

    def infer(self, audio_input):
        raise Exception('The infer function must be overriden by the agent')

    def test(self, save_incorrect_images=True):
        accuracy_list = list()
        if save_incorrect_images:
            incorrect_image_list = list()
        for _ in trange(self.data.test.num_examples//self.batch_size, desc="Testing", leave=False, ncols=100):
            images, labels = self.data.test.next_batch(self.batch_size)
            feed_dict = {self.placeholders['image']: images,
                         self.placeholders['label']: labels}

            accuracy, correct = self.sess.run([self.accuracy, self.correct], feed_dict=feed_dict)
            accuracy_list.append(accuracy)
            if save_incorrect_images:
                if not np.all(correct.astype(np.bool)):
                    incorrect_image_list.append(images[(1-correct).astype(np.bool)])

        total_accuracy = np.mean(accuracy_list)
        if save_incorrect_images:
            stacked_images = np.concatenate(incorrect_image_list, 0)
            incorrect_images = np.reshape(stacked_images, [-1, 28, 28])
            utils.save_mnist_as_image(incorrect_images, "{}/incorrect_images".format(self.result_dir))

        print("Test accuracy: {}".format(total_accuracy))
        return total_accuracy

    def learn_from_epoch(self):
        for _ in trange(self.data.train.num_examples//self.batch_size, desc="Epoch ({} of {}) Iterations".format(self.epoch_id + 1, self.max_train_epochs), leave=False, ncols=100):
            # Get batch
            images, labels = self.data.train.next_batch(self.batch_size)
            feed_dict = {self.placeholders['image']: images,
                         self.placeholders['label']: labels}

            global_step = self.sess.run(self.global_step)  # TODO - add condition for max iteration or limit only by max epoch in main

            op_list = [self.train_op]

            train_summary_now = self.train_summary_every > 0 and global_step % self.train_summary_every == 0
            if train_summary_now:
                op_list.append(self.train_summary)

            train_out = self.sess.run([self.train_op, self.train_summary], feed_dict=feed_dict)

            # Add to tensorboard
            if train_summary_now:
                self.train_summary_writer.add_summary(train_out[1], global_step)

            validate_now = self.validation_summary_every > 0 and global_step % self.validation_summary_every == 0
            if validate_now:
                # Only run 1 batch of validation data while training for speed; can do full test later
                validation_images, validation_labels = self.data.validation.next_batch(self.batch_size)
                validation_feed_dict = {self.placeholders['image']: validation_images,
                                        self.placeholders['label']: validation_labels}

                validation_summary = self.sess.run(self.validation_summary, feed_dict=validation_feed_dict)

                self.validation_summary_writer.add_summary(validation_summary, global_step)