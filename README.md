# Matrix Capsules with EM Routing

TensorFlow Implementation of [Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb).

## Requirements

Created and tested using:
- Python 3.5
- TensorFlow 1.4

#### Packages

- tensorflow
- numpy
- tqdm (This requirement may be removed by removing the tqdm wrapper around mnist_batch in utils.save_mnist_as_image() and changing trange() to range() in CapsNetEMModel.learn_from_epoch())
- matplotlib (Only required for producing additional results with utils.save_mnist_as_image(), the model may be used without) 

```commandline
pip install -r requirements.txt
```

Or for GPU TensorFlow:

```commandline
pip install -r requirements-gpu.txt
```

## Usage

To train a model on MNIST with default parameters:

```commandline
python main.py
```

To test a trained model:

```commandline
python main.py --test True --result_dir <result_dir>
```

Additional command line arguments are detailed in main.py.

Capsule layers have been implemented in the style of tf.layers. These can be found in capsule_layers.py if you would 
like to experiment with custom architectures. 

## Results

Working on this

## Notes

**Matrix Capsules with EM Routing paper**

Some points that caused confusion during this implementation:

**_Beta_a and Beta_v parameters_**

As per the explanation in the [OpenReview comments](https://openreview.net/forum?id=HJWLfGWRb) these are per higher 
level capsule type and are therefore vectors. The initial value of these seems to have a reasonable impact on the speed
of the early stage of training, although initial values are not mentioned in the paper.

**_Inverse temperature parameters_**

The initial inverse temperature parameters and schedule appear to have a significant effect on the training of the model 
(as the activation and activation gradient magnitudes both depend on how saturated the sigmoid function in the E-step is, 
which is controlled by the inverse temperature parameter) however these values are also not mentioned in the paper.

**_Padding_**

The paper seems to imply 'SAME' padding for convolutional layers given that the bottom of section 4 suggests that corner
capsules will only receive "one feedback per capsule type", although this is not explicitly mentioned.

**_Sum over capsules in the E-step_**

This has been slightly clarified in the new (05/01/18) version of the paper but I think it is still not explicit.
My interpretation is that this is the sum over output capsules for which the input capsule is within the receptive
field. For a 1D image with one 1D capsule and a patch size of 2 this would look like (ignoring element wise operations
for the purposes of demonstrating only the effect of shape changing operations):

*Input image (indexed by i in paper)*

[[1, 2, 3, 4]]

*Patches as returned by extract_image_patches_nd (indexed by j in paper)*

[[1, 2],

 [2, 3],
 
 [3, 4]] 

*Patches in correct position*

[[1, 2, x, x],

 [x, 2, 3, x],
 
 [x, x, 3, 4]]
 
*Correct sum*

[[1, 4, 6, 4]]

This implementation computes this by first converting to a sparse tensor before computing the sum.

*Incorrect sum*

From a brief look at other implementations it looks like some are computing the sum over the patches as in the
second step, without taking into account receptive fields, and would therefore get:

[[6, 9]]


**TensorFlow limitations**

This implementation required a fair number of workarounds for limitations in TensorFlow e.g. tf.sparse_reduce_sum and 
tf.gather_nd do not support tensors > 5D, however for convolutional capsules it is convenient to deal with 9D tensors
as this preserves the 2D structure of the capsules between layers. Also this implementation would have been made easier 
if there was better support for sparse tensors. It would be useful if there was a version of tf.extract_image_patches
that returned a sparse tensor. It would also be useful if there was a proper sparse version of reduce_logsumexp.
Future implementations of capsule networks would definitely benefit from new capsule specific TensorFlow ops. 

### TODO

- [ ] Get convolutional capsule layers working for strides > 1
- [ ] Test convolutional capsule layers with padding 'SAME'
- [ ] Get working with SmallNORB dataset
- [ ] Sort out license
- [ ] Implement hyperparameter search
- [ ] Implement easy resume training
- [ ] Improve safe divide so that it will always produce a representable output

## Reference
1. [Matrix Capsules with EM Routing paper](https://openreview.net/pdf?id=HJWLfGWRb)

2. [Matrix Capsules with EM Routing OpenReview comments](https://openreview.net/forum?id=HJWLfGWRb) 

3. [Logsumexp trick used for stability in the E-step of EM routing](https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow/issues/10)

4. [Project structure](https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3)


