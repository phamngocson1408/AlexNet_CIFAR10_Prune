import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
print(train_images.shape)
sample_to_predict = tf.image.resize(test_images[1], (277, 277))
print(sample_to_predict.shape)
sample_to_predict = tf.reshape(sample_to_predict, (-1, 277, 277, 3))
print(sample_to_predict.shape)
#resized_train_images = tf.image.resize(train_images, (277, 277))
resized_test_images = tf.image.resize(test_images, (277, 277))
#print(resized_test_images.shape)

CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(277, 277, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])

model.load_weights('saved_weights/')

"""
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.optimizers.SGD(learning_rate=0.001),
    metrics=['accuracy'],
)

model.fit(resized_train_images, train_labels, batch_size=32, epochs=50)
model.save_weights('saved_weights/')
"""

import tensorflow_model_optimization as tfmot
import numpy as np

# Compute end step to finish pruning after 2 epochs.
batch_size = 32
epochs = 2
"""
end_step = np.ceil(resized_train_images.shape[0] / batch_size).astype(np.int32) * epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.5,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=end_step)
}
"""

# Helper function uses `prune_low_magnitude` to make only the
# Dense layers train with pruning.
def apply_pruning_to_conv(layer):
  if isinstance(layer, tf.keras.layers.Conv2D):
    return tfmot.sparsity.keras.prune_low_magnitude(layer)
  return layer

# Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense`
# to the layers of the model.
model_for_pruning = tf.keras.models.clone_model(
    model,
    clone_function=apply_pruning_to_conv,
)
model_for_pruning.summary()

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep()
]

model_for_pruning.load_weights('saved_pruned_weights/')

model_for_pruning.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.optimizers.SGD(learning_rate=0.001),
    metrics=['accuracy'],
)

#model_for_pruning.fit(resized_train_images, train_labels, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
#model_for_pruning.save_weights('saved_pruned_weights/')

model_for_pruning.evaluate(resized_test_images, test_labels)

for layer in model_for_pruning.layers:
    zero_weights = tf.size(layer.weights[0]).numpy() - tf.math.count_nonzero(layer.weights[0]).numpy()
    weight_sparsity = zero_weights / tf.size(layer.weights[0]).numpy() * 100
    print("Weight sparsity of layer", layer.name, "is:", weight_sparsity)
#    print(layer.weights.shape)

from keras import backend as K

inp = model_for_pruning.input                                           # input placeholder
outputs = [layer.output for layer in model_for_pruning.layers]          # all layer outputs
functors = [K.function([inp], [out]) for out in outputs]                # evaluation functions

# Testing
#layer_outs = functor([sample_to_predict, 1.])
layer_outs = [func([sample_to_predict]) for func in functors]

layer_idx = 0
for layer_out in layer_outs:
    layer_idx += 1
    zero_neurons = tf.size(layer_out).numpy() - tf.math.count_nonzero(layer_out).numpy()
    neuron_sparsity = zero_neurons / tf.size(layer_out).numpy() * 100
    print("Neuron sparsity of layer", layer_idx, "is:", neuron_sparsity)

def visualize_conv_layer(layer_idx):
    layer_output = model_for_pruning.layers[layer_idx].output

    intermediate_model = tf.keras.models.Model(inputs=model_for_pruning.input,
                                               outputs=layer_output)

    intermediate_prediction = intermediate_model.predict(sample_to_predict)
    zero_outputs = tf.size(intermediate_prediction).numpy() - tf.math.count_nonzero(intermediate_prediction).numpy()
    sparsity = zero_outputs / tf.size(intermediate_prediction).numpy() * 100
    print("Output sparsity of layer[", layer_idx, "] is:", sparsity)

#visualize_conv_layer(3)
