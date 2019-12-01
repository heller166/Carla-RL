import os
import sys
import settings
import pickle
import time
import numpy as np
from sources import models
from dataclasses import dataclass

# Try to mute and then load Tensorflow and Keras
# Muting seems to not work lately on Linux in any way
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#stdin = sys.stdin
#sys.stdin = open(os.devnull, 'w')
#stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')
#tf.logging.set_verbosity(tf.logging.ERROR)
from keras.optimizers import Adam
from keras.models import load_model, Model
#sys.stdin = stdin
#sys.stderr = stderr

# Image types
@dataclass
class AGENT_IMAGE_TYPE:
    rgb = 0
    grayscaled = 1
    stacked = 2


# Agent class
class BaseAgent:
    def __init__(self, model_path=False, id=None):

        # Set model path if provided (used when playing, when training we use weights from the trainer)
        self.model_path = model_path

        # Set to show an output from Conv2D layer
        self.show_conv_cam = (id + 1) in settings.CONV_CAM_AGENTS

        # Main model (agent does not use target model)
        self.model = self.create_model(prediction=True)

        # Indicator if currently newest weights are the one agent's model is already using
        self.weights_iteration = 0

        # Set to terminate additional threads on main process exit
        self.terminate = False

    # Load or create a new model (loading a model is being used only when playing or by trainer class that inherits from agent)
    def create_model(self, prediction=False):

        # If there is a patht to the model set, load model
        if self.model_path:
            model = load_model(self.model_path)
            self._extract_model_info(model)

            # If we show convcam, we need additional output from given layer
            if prediction and self.show_conv_cam:
                model = Model(model.input, [model.output, model.layers[self.convcam_layer].output])

            return model

        # Create the model
        model_base = getattr(models, 'model_base_' + settings.MODEL_BASE)(
            (settings.IMG_HEIGHT, settings.IMG_WIDTH,
             1 if settings.AGENT_IMG_TYPE == AGENT_IMAGE_TYPE.grayscaled else 3))
        model = getattr(models, 'model_head_' + settings.MODEL_HEAD)(*model_base, outputs=len(settings.ACTIONS),
                                                                     model_settings=settings.MODEL_SETTINGS)

        self._extract_model_info(model)

        # We need to compile model only for training purposes, agents do not compile their models
        if not prediction:
            self.compile_model(model=model, lr=settings.OPTIMIZER_LEARNING_RATE, decay=settings.OPTIMIZER_DECAY)
        # If we show convcam, we need additional output from given layer
        elif self.show_conv_cam:
            model = Model(model.input, [model.output, model.layers[self.convcam_layer].output])

        return model

    def _extract_model_info(self, model):
        # Create strings that can be used in model name (automatically extracted from the model)
        model_architecture = []
        cnn_kernels = []
        last_conv_layer = None
        for index, layer in enumerate(model.layers):
            layer_name = layer.__class__.__name__.split('_')[-1]
            if layer_name == 'Activation' or layer_name == 'InputLayer':
                if layer_name == 'Activation' and settings.CONV_CAM_LAYER == 'auto_act' and index == last_conv_layer + 1:
                    last_conv_layer += 1
                continue

            if layer_name.startswith('Conv'):
                cnn_kernels.append(str(layer.filters))
                last_conv_layer = index

            if layer_name == 'Dropout':
                layer_name = 'DRopout'
            layer_acronym = ''.join(filter(str.isupper, layer_name.replace('1D', '').replace('2D', '').replace('3D', '')))
            if hasattr(layer, 'filters'):
                layer_acronym += str(layer.filters)
            elif hasattr(layer, 'units'):
                layer_acronym += str(layer.units)

            model_architecture.append(layer_acronym)

        model_architecture = '-'.join(model_architecture)
        cnn_kernels = '-'.join(cnn_kernels)

        settings.MODEL_NAME = settings.MODEL_NAME.replace('#MODEL_ARCHITECTURE#', model_architecture)
        settings.MODEL_NAME = settings.MODEL_NAME.replace('#CNN_KERNELS#', cnn_kernels)
        self.convcam_layer = last_conv_layer if settings.CONV_CAM_LAYER in ['auto', 'auto_act'] else settings.CONV_CAM_LAYER

    # Compiles a model given learning rate and decay factor
    def compile_model(self, model, lr, decay):
        model.compile(loss="mse", optimizer=Adam(lr=lr, decay=decay), metrics=['accuracy'])

    # Gets weight's object and returns a numpy array of weights
    def decode_weights(self, weights):
        return pickle.loads(weights.raw)

    # Updates model's weights
    def update_weights(self):

        # Decode weights
        model_weights = self.decode_weights(self.weights)

        # And update them
        self.model.set_weights(model_weights)

    # To be ran in a separate thread
    # Monitors if there are new weights being saved by trainer and updates them
    def update_weights_in_loop(self):

        # Do not update in loop?
        if settings.UPDATE_WEIGHTS_EVERY <= 0:
            return

        while True:

            # If process finishes it's work, exit this thread
            if self.terminate:
                return

            # If trainer's weights are in a newer revision - save them then update
            if self.trainer_weights_iteration.value >= self.weights_iteration + settings.UPDATE_WEIGHTS_EVERY:
                self.weights_iteration = self.trainer_weights_iteration.value + settings.UPDATE_WEIGHTS_EVERY
                self.update_weights()
            else:
                time.sleep(0.001)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):

        # Create array of inputs and add normalized image
        Xs = [np.array(state[0]).reshape(-1, *state[0].shape)/255]

        # Additional inputs?
        if 'kmh' in settings.AGENT_ADDITIONAL_DATA:
            Xs.append((np.array([state[1]]).reshape(-1, 1) - 50) / 50)

        # Predict and return (return additional output when convcam is being used)
        prediction = self.model.predict(Xs)
        if self.show_conv_cam:
            return [prediction[0][0], prediction[1][0]]
        else:
            return [prediction[0]]

    # Prepares received image (observation space)
    def prepare_image(self, image, create=False):

        # If RGB image - return it
        if settings.AGENT_IMG_TYPE == AGENT_IMAGE_TYPE.rgb:
            return image

        # If grayscaled - make grayscale, convert type and add last dimension
        elif settings.AGENT_IMG_TYPE == AGENT_IMAGE_TYPE.grayscaled:
            return np.expand_dims(np.dot(image, [0.299, 0.587, 0.114]).astype('uint8'), -1)

        # If stacked, we stack last 3 consecutive frames
        elif settings.AGENT_IMG_TYPE == AGENT_IMAGE_TYPE.stacked:

            # If environment got reset - create new image
            if create:
                image = np.dot(image, [0.299, 0.587, 0.114]).astype('uint8')
                self.image = np.stack([image, image, image], axis=-1)

            # If step - move frames by 1 and replace first one with a new image
            else:
                self.image = np.roll(self.image, 1, -1)
                self.image[..., 0] = np.dot(image, [0.299, 0.587, 0.114]).astype('uint8')
            return self.image

