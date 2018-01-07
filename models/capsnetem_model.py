from models import BaseModel
from models import utils
import numpy as np
from tqdm import trange
import tensorflow as tf

from models.capsule_layers import primarycaps_layer, convcaps_layer, classcaps_layer
from models.capsule_ops import spread_loss


class CapsNetEMModel(BaseModel):
    def set_model_props(self, config):
        # Spread Loss
        self.initial_margin = self.config['initial_margin']
        self.final_margin = self.config['final_margin']
        self.margin_decay_steps = self.config['margin_decay_steps']

        # Set up parameter dicts
        self.relu_conv1_params = {'kernel_size': self.config['relu_conv1_kernel_size'],
                                  'filters': self.config['relu_conv1_filters'],
                                  'strides': self.config['relu_conv1_stride']}

        self.primarycaps_params = {'out_capsules': self.config['primarycaps_out_capsules'],
                                   'pose_size': self.config['pose_size']}

        self.convcaps1_params = {'out_capsules': self.config['convcaps1_out_capsules'],
                                 'kernel_size': self.config['convcaps1_kernel_size'],
                                 'strides': self.config['convcaps1_strides'],
                                 'padding': self.config['convcaps1_padding'],
                                 'n_routing_iterations': self.config['convcaps1_n_routing_iterations'],
                                 'init_beta_v': self.config['convcaps1_init_beta_v'],
                                 'init_beta_a': self.config['convcaps1_init_beta_a'],
                                 'init_inverse_temp': self.config['convcaps1_init_inverse_temp'],
                                 'final_inverse_temp': self.config['convcaps1_final_inverse_temp']}

        self.convcaps2_params = {'out_capsules': self.config['convcaps2_out_capsules'],
                                 'kernel_size': self.config['convcaps2_kernel_size'],
                                 'strides': self.config['convcaps2_strides'],
                                 'padding': self.config['convcaps2_padding'],
                                 'init_beta_v': self.config['convcaps2_init_beta_v'],
                                 'init_beta_a': self.config['convcaps2_init_beta_a'],
                                 'n_routing_iterations': self.config['convcaps2_n_routing_iterations'],
                                 'init_inverse_temp': self.config['convcaps2_init_inverse_temp'],
                                 'final_inverse_temp': self.config['convcaps2_final_inverse_temp']}

        self.classcaps_params = {'n_classes': self.config['n_classes'],
                                 'n_routing_iterations': self.config['classcaps_n_routing_iterations'],
                                 'init_beta_v': self.config['classcaps_init_beta_v'],
                                 'init_beta_a': self.config['classcaps_init_beta_a'],
                                 'init_inverse_temp': self.config['classcaps_init_inverse_temp'],
                                 'final_inverse_temp': self.config['classcaps_final_inverse_temp']}

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
            tf.summary.scalar('margin', self.margin)

            self.spread_loss_params = {'margin': self.margin}

            # Initalise summaries dict - using dict so that we can merge only select summaries; don't want image summaries all
            # the time
            self.summaries = {}

            # Reshape flattened image tensor to 2D
            images = tf.reshape(self.placeholders['image'], [-1, 28, 28, 1])
            # summaries['images'] = tf.summary.image('input_images', images)
            tf.summary.image('input_images', images, max_outputs=1)

            # Create ReLU Conv1 de-rendering layer
            with tf.variable_scope('relu_conv1'):
                relu_conv1_out = tf.layers.conv2d(images, **self.relu_conv1_params, activation=tf.nn.relu,
                                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.5))

            # Create PrimaryCaps layer
            primarycaps_pose, primarycaps_activation = primarycaps_layer(relu_conv1_out, **self.primarycaps_params)
            tf.summary.image('primarycaps_activation_image_cap_0', tf.expand_dims(primarycaps_activation[:, :, :, 0], 3), max_outputs=1)
            tf.summary.image('primarycaps_activation_image_cap_1', tf.expand_dims(primarycaps_activation[:, :, :, 1], 3), max_outputs=1)
            tf.summary.image('primarycaps_activation_image_cap_2', tf.expand_dims(primarycaps_activation[:, :, :, 2], 3), max_outputs=1)

            # Create ConvCaps1 layer
            convcaps1_pose, convcaps1_activation = convcaps_layer(primarycaps_pose, primarycaps_activation, **self.convcaps1_params)
            tf.summary.image('convcaps1_activation_image_cap_0', tf.expand_dims(convcaps1_activation[:, :, :, 0], 3), max_outputs=1)
            tf.summary.image('convcaps1_activation_image_cap_1', tf.expand_dims(convcaps1_activation[:, :, :, 1], 3), max_outputs=1)
            tf.summary.image('convcaps1_activation_image_cap_2', tf.expand_dims(convcaps1_activation[:, :, :, 2], 3), max_outputs=1)

            # Create ConvCaps2 layer
            convcaps2_pose, convcaps2_activation = convcaps_layer(convcaps1_pose, convcaps1_activation, **self.convcaps2_params)
            #convcaps2_pose, convcaps2_activation = convcaps_layer(primarycaps_pose, primarycaps_activation, **self.convcaps2_params)
            tf.summary.image('convcaps2_activation_image_cap_0', tf.expand_dims(convcaps2_activation[:, :, :, 0], 3), max_outputs=1)
            tf.summary.image('convcaps2_activation_image_cap_1', tf.expand_dims(convcaps2_activation[:, :, :, 1], 3), max_outputs=1)
            tf.summary.image('convcaps2_activation_image_cap_2', tf.expand_dims(convcaps2_activation[:, :, :, 2], 3), max_outputs=1)

            # Create Class Capsules layer
            classcaps_pose, classcaps_activation = classcaps_layer(convcaps2_pose, convcaps2_activation, **self.classcaps_params)
            # classcaps_pose, classcaps_activation = classcaps_layer(primarycaps_pose, primarycaps_activation, **classcaps_params)

            # Create spread loss
            self.loss = spread_loss(classcaps_activation, self.placeholders['label'], **self.spread_loss_params)

            # Get predictions, accuracy, correct and summaries
            with tf.name_scope("accuracy"):
                self.predictions = tf.argmax(classcaps_activation, axis=1)
                labels = tf.argmax(self.placeholders['label'], axis=1)
                self.correct = tf.cast(tf.equal(labels, self.predictions), tf.int32)
                self.accuracy = tf.reduce_sum(self.correct) / tf.shape(self.correct)[0]  # reduce_mean not working here for some reason
                self.summaries['accuracy'] = tf.summary.scalar('accuracy', self.accuracy)
            self.summaries['loss'] = tf.summary.scalar('loss', self.loss)

            tf.summary.histogram('primarycaps_activation', primarycaps_activation)
            tf.summary.histogram('convcaps1_activation', convcaps1_activation)
            tf.summary.histogram('convcaps2_activation', convcaps2_activation)
            tf.summary.histogram('classcaps_activation', classcaps_activation)
            tf.summary.histogram('correct', self.correct)

            # Define optimiser
            self.optim = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optim.minimize(self.loss, global_step=self.global_step)

            # Set up summaries
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
        for _ in trange(self.data.train.num_examples//self.batch_size,
                        desc="Epoch ({} of {}) Iterations".format(self.epoch_id + 1, self.max_train_epochs),
                        leave=False, ncols=100):
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
