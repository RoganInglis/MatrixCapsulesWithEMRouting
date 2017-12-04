from models import BaseModel
from models import utils
import numpy as np
from tqdm import trange
import tensorflow as tf


class CapsNetEMModel(BaseModel):
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
        with graph.as_default():
            # Create placeholders
            self.placeholders = {'image': tf.placeholder(tf.float32, [None, self.image_dim], name='image'),
                                 'label': tf.placeholder(tf.int32, [None, self.n_classes], name='label')}

            # Define main model graph
            self.loss, self.predictions, self.accuracy, self.correct, self.summaries = utils.build_capsnetem_graph(
                                                                                                   self.placeholders,
                                                                                                   image_dim=self.image_dim)

            # Define optimiser
            self.optim = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optim.minimize(self.loss, global_step=self.global_step)

            # Set up summaries
            self.train_summary = tf.summary.merge([self.summaries['accuracy'],
                                                   self.summaries['loss'],
                                                   *self.summaries['general']])
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