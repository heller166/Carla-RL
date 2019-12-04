import os
import sys
import settings
import pickle
import time
import numpy as np
from sources import BaseAgent
from sources import CarlaEnv, STOP, models, ACTIONS_NAMES
from collections import deque
from threading import Thread
from dataclasses import dataclass
import cv2
import imageio

# Try to mute and then load Tensorflow and Keras
# Muting seems to not work lately on Linux in any way
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#stdin = sys.stdin
#sys.stdin = open(os.devnull, 'w')
#stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
import keras.backend.tensorflow_backend as backend
from keras.optimizers import Adam
from keras.models import load_model, Model
#sys.stdin = stdin
#sys.stderr = stderr

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


# Agent class
class ARTDDPGAgent(BaseAgent):
    def __init__(self, model_path=False, id=None):

        # Set model path if provided (used when playing, when training we use weights from the trainer)
        self.model_path = model_path

        # Set to show an output from Conv2D layer
        self.show_conv_cam = (id + 1) in settings.CONV_CAM_AGENTS

        # Main model (agent does not use target model)
        self.actor = self.create_model(prediction=True, is_actor=True)
        self.critic = self.create_model(prediction=True, is_actor=False)

        self.noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(len(settings.ACTIONS)))

        # Indicator if currently newest weights are the one agent's model is already using
        self.weights_iteration = 0

        # Set to terminate additional threads on main process exit
        self.terminate = False

    # Load or create a new model (loading a model is being used only when playing or by trainer class that inherits from agent)
    def create_model(self, prediction=False, is_actor=False):

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
            (settings.IMG_HEIGHT, settings.IMG_WIDTH, 3))
        model = getattr(models, 'model_head_' + settings.MODEL_HEAD_ACTOR
                                if is_actor else 'model_head_' + settings.MODEL_HEAD_CRITIC)(
            *model_base, outputs=len(settings.ACTIONS) if is_actor else 1, model_settings=settings.MODEL_SETTINGS)

        self._extract_model_info(model)

        # We need to compile model only for training purposes, agents do not compile their models
        if not prediction:
            self.compile_model(model=model, lr=settings.OPTIMIZER_LEARNING_RATE, decay=settings.OPTIMIZER_DECAY)
        # If we show convcam, we need additional output from given layer
        elif self.show_conv_cam:
            model = Model(model.input, [model.output, model.layers[self.convcam_layer].output])

        return model

    # Compiles a model given learning rate and decay factor
    def compile_model(self, model, lr, decay):
        model.compile(loss="mse", optimizer=Adam(lr=lr, decay=decay), metrics=['accuracy'])

    # Gets weight's object and returns a numpy array of weights
    def decode_weights(self, weights):
        return pickle.loads(weights.raw)

    # Updates model's weights
    def update_weights(self):

        # Decode weights
        critic_weights = self.decode_weights(self.weights_critic)
        actor_weights = self.decode_weights(self.weights_actor)

        # And update them
        self.critic.set_weights(critic_weights)
        self.actor.set_weights(actor_weights)

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

    def get_actions(self, state):

        # Create array of inputs and add normalized image
        Xs = [np.array(state[0]).reshape(-1, *state[0].shape)/255]

        # Predict and return (return additional output when convcam is being used)
        return self.actor.predict(Xs)

    # Queries main network for Q values given current observation space (environment state)
    def get_q(self, state, actions):
    #TODO: Check the output of both of this methods.
    #TODO: the output is kind of confusing. When we have more clearly understoon the form we should double check
        # Create array of inputs and add normalized image
        Xs = [np.array(state[0]).reshape(-1, *state[0].shape)/255, actions]

        # Predict and return (return additional output when convcam is being used)
        return self.critic.predict(Xs)


# Image types
@dataclass
class AGENT_IMAGE_TYPE:
    rgb = 0
    grayscaled = 1
    stacked = 2


# Agent states
@dataclass
class AGENT_STATE:
    starting = 0
    playing = 1
    restarting = 2
    finished = 3
    error = 4
    paused = 5


# Agent state messages
AGENT_STATE_MESSAGE = {
    0: 'STARTING',
    1: 'PLAYING',
    2: 'RESTARING',
    3: 'FINISHED',
    4: 'ERROR',
    5: 'PAUSED',
}


# Runs agent process
def run(id, carla_instance, stop, pause, episode, epsilon, show_preview, weights_actor, weights_critic, weights_iteration, transitions,
        tensorboard_stats, agent_stats, carla_frametimes, seconds_per_episode):

    # Set GPU used for an agent
    if settings.AGENT_GPU is not None and type(settings.AGENT_GPU) == int:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.AGENT_GPU)
    elif settings.AGENT_GPU is not None and type(settings.AGENT_GPU) == list:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.AGENT_GPU[id])

    # Agent memory fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings.AGENT_MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Create an agent, set weights from object shared by trainer
    agent = ARTDDPGAgent(id=id)
    agent.weights_critic = weights_critic
    agent.weights_actor = weights_actor
    agent.trainer_weights_iteration = weights_iteration
    agent.update_weights()

    # Wait for Carla to be ready
    while True:

        # If stop command - exit
        if stop.value == STOP.stopping:
            agent_stats[0] = AGENT_STATE.finished
            return

        # Try to create environment. If fails, try again
        try:
            env = CarlaEnv(carla_instance, seconds_per_episode)
            break
        except:
            agent_stats[0] = AGENT_STATE.error
            time.sleep(1)

    agent_stats[0] = AGENT_STATE.starting

    # Set shared array for carla FPS measurement
    env.frametimes = carla_frametimes

    # Set a list for agent FPS measurement
    fps_counter = deque(maxlen=60)

    # Create a separate thread for weights update so it won't stop predicting on weights update
    weight_updater = Thread(target=agent.update_weights_in_loop, daemon=True)
    weight_updater.start()

    # Predict once on any data to initialize predictions (won't stop episode during first call)
    actions = agent.get_actions([np.ones((env.im_height, env.im_width,
                                1 if settings.AGENT_IMG_TYPE == AGENT_IMAGE_TYPE.grayscaled else 3)), [0]])
    agent.get_q([np.ones((env.im_height, env.im_width,
                           1 if settings.AGENT_IMG_TYPE == AGENT_IMAGE_TYPE.grayscaled else 3)), [0]], actions)

    agent_stats[0] = AGENT_STATE.playing

    gif_images = []

    # Play as long as there is no 'stop' command being issued
    while stop.value != STOP.stopping:

        # if simulator restarts - do not try to connect yet
        if stop.value == STOP.restarting_carla_simulator:
            time.sleep(0.1)
            continue

        # Carla want's agent to pause (world change) -pause and inform Carla about that
        if pause.value == 1:
            pause.value = 2
            agent_stats[0] = AGENT_STATE.paused

        # Wait for Carla to release pause
        if pause.value == 2:
            time.sleep(0.1)
            continue

        # Pause lock released, reconnect and play
        if pause.value == 3:
            pause.value = 0
            agent_stats[0] = AGENT_STATE.starting
            try:
                env.destroy_agents()
            except:
                pass
            try:
                env = CarlaEnv(carla_instance, seconds_per_episode)
                env.frametimes = carla_frametimes
            except:
                pass

            # Sleep for a second and restart episode
            time.sleep(1)
            continue


        # Restarting episode - reset episode reward, step number and episode predicted qs list
        episode_reward = 0
        step = 1
        predicted_qs = [[] for _ in range(env.action_space_size + 1)]
        predicted_actions = [0 for _ in range(env.action_space_size + 1)]

        # Reset environment
        try:

            # Destroy agents if there are any and restart environment
            env.destroy_agents()
            current_state = env.reset()

            # Prepare image accordingly to settings
            current_state[0] = agent.prepare_image(current_state[0], create=True)
        except:

            # When above fails, set error state...
            agent_stats[0] = AGENT_STATE.error

            # ...and try to create a new environment (necessary after Carla restart)
            try:
                env.destroy_agents()
            except:
                pass
            try:
                env = CarlaEnv(carla_instance, seconds_per_episode)
                env.frametimes = carla_frametimes
            except:
                pass

            # Sleep for a second and restart episode
            time.sleep(1)
            continue

        # Update weights if updates on episode restart
        if settings.UPDATE_WEIGHTS_EVERY == 0:
            agent.update_weights()

        agent_stats[0] = AGENT_STATE.playing

        # Clear collision history at this point
        env.collision_hist = []

        # Reset 'done' flag and set initial values for episode duration measurement
        done = False
        episode_start = episode_end = time.time()

        # For synced mode
        last_processed_cam_update = 0

        # Reset min and max values for Convcam
        conv_min = None
        conv_max = None

        # Steps
        while True:

            # Set step number in agent stats
            agent_stats[1] = step

            # Start measuring step time
            step_start = time.time()

            # If synced more - wait for updated frame, but no longer than a second
            if settings.AGENT_SYNCED:
                wait_for_frame_start = time.time()
                while True:
                    if env.last_cam_update > last_processed_cam_update:
                        last_processed_cam_update = env.last_cam_update
                        break
                    if time.time() > wait_for_frame_start + 1:
                        break
                    time.sleep(0.001)

            # Get action from predicted Q values
            actions = np.clip(agent.get_actions(current_state)[0] + agent.noise(), -1, 1)

            # Log Q values for episode mean
            # TODO: Check how to get these logs squared up
            #for i in range(env.action_space_size):
            #    predicted_qs[0].append(qs[0][i])
            #    predicted_qs[i + 1].append(qs[0][i])
            #predicted_actions[0] += 1
            #predicted_actions[action + 1] += 1

            # Convcam
            #TODO: understand what convcam is and why we need it
            #if agent.show_conv_cam:

            #    # Calculate min and max values and add them weighted - stabilizes image flickering
            #    conv_min = np.min(qs[1]) if conv_min is None else 0.8 * conv_min + 0.2 * np.min(qs[1])
            #    conv_max = np.max(qs[1]) if conv_max is None else 0.8 * conv_max + 0.2 * np.max(qs[1])

            #    # Normalize to 0..255
            #    conv_preview = ((qs[1] - conv_min) * 255 / (conv_max - conv_min)).astype(np.uint8)

            #    # Swap axes and reshape to format output image
            #    conv_preview = np.moveaxis(conv_preview, 1, 2)
            #    conv_preview = conv_preview.reshape((conv_preview.shape[0], conv_preview.shape[1] * conv_preview.shape[2]))

            #    # Find where to "wrap" wide image
            #    i = 1
            #    while not (conv_preview.shape[1] / qs[1].shape[1]) % (i * i):
            #        i *= 2
            #    i //= 2

            #    # Wrap image
            #    conv_preview_reorganized = np.zeros((conv_preview.shape[0] * i, conv_preview.shape[1] // i), dtype=np.uint8)
            #    for start in range(i):
            #        conv_preview_reorganized[start * conv_preview.shape[0]:(start + 1) * conv_preview.shape[0], 0:conv_preview.shape[1] // i] = conv_preview[:, (conv_preview.shape[1] // i) * start:(conv_preview.shape[1] // i) * (start + 1)]

            #    # Show image
            #    cv2.imshow(f'Agent {id + 1} - Convcam', conv_preview_reorganized)
            #    cv2.waitKey(1)

            # Try to step environment, break episode on error
            try:
                new_state, reward, done, _ = env.step(actions)
            except:
                agent_stats[0] = AGENT_STATE.error
                time.sleep(1)
                break

            # Show a preview if set (env)
            if show_preview[0] == 1:
                if episode.value % 1000 == 0:
                    gif_images.append(new_state[0])
                cv2.imshow(f'Agent {id+1} - preview', new_state[0])
                cv2.waitKey(1)
                env.preview_camera_enabled = False

            # Prepare observation space accordingly to settings
            new_state[0] = agent.prepare_image(new_state[0])

            # Show a preview if set (agent)
            if show_preview[0] == 2:
                cv2.imshow(f'Agent {id+1} - preview', new_state[0])
                cv2.waitKey(1)
                env.preview_camera_enabled = False

            # Show preview if set ("above the car" camera)
            if show_preview[0] >= 10 or show_preview[0] == 3:
                if show_preview[0] == 3:
                    env.preview_camera_enabled = show_preview[1:]
                else:
                    env.preview_camera_enabled = settings.PREVIEW_CAMERA_RES[10 - int(show_preview[0])]
                if env.preview_camera is not None:
                    cv2.imshow(f'Agent {id+1} - preview', env.preview_camera)
                    cv2.waitKey(1)

            # Disable preview?
            # try ... except necessary on linux as .getWindowProperty() might error instead of returning -1 value
            try:
                if not show_preview[0] and cv2.getWindowProperty(f'Agent {id+1} - preview', 0) >= 0:
                    cv2.destroyWindow(f'Agent {id + 1} - preview')
                    env.preview_camera_enabled = False
            except:
                pass

            # Count episode reward
            episode_reward += reward

            # Put transition into a shared Queue for trainer
            if settings.AGENT_IMG_TYPE != AGENT_IMAGE_TYPE.stacked or step >= 3:
                transitions.put_nowait((current_state, actions, reward, new_state, done))

            # Set new state as current state (for next step)
            current_state = new_state

            # If 'done' flag from environment is set - end of an episode
            if done:
                episode_end = time.time()
                break

            # A bit overthinked (maybe) way to keep stable agent FPS as long as possible
            # It works based on timeframes and number of frame agent should be at
            # So if, for a while, agent plays slower than desired FPS, it allows it
            # to play a bit faster to keep desired FPS
            # If, for any reason, player constantly plays slower, this code part
            # won't be able to do anything about that

            # Time difference from episode start to the point episode should end
            time_diff = episode_start + step/settings.EPISODE_FPS - time.time()

            # Time difference from step start to the point step should end
            time_diff2 = step_start + 1/settings.EPISODE_FPS - time.time()

            # Sleep to the point this step should end (not not shorter than 5ms
            # (Carla does not like actions to be sent too fast - a case when epsilon is high
            # and step takes almost no time - prediction on a model is what's the very slowest part here)
            if time_diff > 0:
                time.sleep(min(0.125, time_diff))
            elif time_diff2 > 0:
                time.sleep(min(0.125, time_diff2))

            # Increase step number for next step
            step += 1

            # Frame time, also whole episode time (including sleeping, for agent FPS measurement)
            # Adds to a deque and...
            frame_time = time.time() - step_start
            fps_counter.append(frame_time)

            # ... counts mean value from last n frames
            agent_stats[2] = len(fps_counter)/sum(fps_counter)

        # Episode ended, remove actors
        try:
            env.destroy_agents()
        except:
            pass

        # We need to check for 'done' flag as episode might end because of an error
        if done:
            if episode.value % 1000 == 0:
                imageio.mimsave('gifs/ddpg_{}.gif'.format(episode.value), gif_images, fps=20)
            gif_images = []

            # Duration of current episode
            episode_time = episode_end - episode_start

            # Average FPS of an episode - for statistics and reward weight
            average_fps = step / episode_time

            # When epsilon goes down, more and more actions are being taken by model prediction
            # That causes less rewards being collected during fixed episode time so episode
            # reward is going to be lower. Simple (but not ideal) way to get reward that means
            # something is to weight episode reward by a factor of mean episode FPS and desired one
            # The lower FPS the factor is bigger
            reward_factor = settings.EPISODE_FPS / average_fps
            episode_reward_weighted = (episode_reward - reward) * reward_factor + reward

            # Average of predicted Q value for each action (for statistics)
            avg_predicted_qs = []
            for i in range(env.action_space_size + 1):
                if len(predicted_qs[i]):
                    avg_predicted_qs.append(sum(predicted_qs[i])/len(predicted_qs[i]))
                    avg_predicted_qs.append(np.std(predicted_qs[i]))
                    avg_predicted_qs.append(100 * predicted_actions[i] / predicted_actions[0])
                else:
                    avg_predicted_qs.append(-10**6)
                    avg_predicted_qs.append(-10**6)
                    avg_predicted_qs.append(-10**6)


            # Add current episode data to the stared Queue
            # += operation is not atomic, we need to use a lock to make sure it increases correctly
            with episode.get_lock():
                episode.value += 1
                tensorboard_stats.put([episode.value, episode_reward, epsilon[0], episode_time, agent_stats[2], episode_reward_weighted] + avg_predicted_qs)

            # Decay epsilon
            # epsilon is an array of 3 elements: current epsilon [0], epsilon decay value [1] and minimal epsilon to heep [2]
            if epsilon[0] > epsilon[2]:
                with epsilon.get_lock():
                    epsilon[0] *= epsilon[1]
                    epsilon[0] = max(epsilon[2], epsilon[0])

        # Set agent state to "restart", step to 0 and FPS to 0
        agent_stats[0] = AGENT_STATE.restarting
        agent_stats[1] = 0
        agent_stats[2] = 0

    # Stop command, exit

    # Set terminate flag for weight updater thread and wait for it to finish
    agent.terminate = True
    weight_updater.join()

    agent_stats[0] = AGENT_STATE.finished

    # It is possible for process to hang here when Queue is full/data is not fully flushed to a Queue/etc
    # Just discard data if there's any, we don't need it anymore
    transitions.cancel_join_thread()
    tensorboard_stats.cancel_join_thread()
    carla_frametimes.cancel_join_thread()


# Play in environment
# TODO: disabling this for now. What to get a simpler version working before thinking about this extra stuff
def play(model_path, pause, console_print_callback):
    pass
#    # Set GPU used for an agent
#    if settings.AGENT_GPU is not None and type(settings.AGENT_GPU) == int:
#        os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.AGENT_GPU)
#    elif settings.AGENT_GPU is not None and type(settings.AGENT_GPU) == list:
#        os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.AGENT_GPU[id])
#
#    # Agent memory fraction
#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings.AGENT_MEMORY_FRACTION)
#    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
#
#    # Create an agent
#    agent = ARTDQNAgent(model_path, id=0)
#
#    # And environment
#    env = CarlaEnv(0, playing=True)
#
#    # Agent and carla FPS counters
#    env.frametimes = deque(maxlen=60)
#    fps_counter = deque(maxlen=60)
#
#    # Predict once on any data to initialize predictions (won't stop episode during first call)
#    agent.get_qs([np.ones((env.im_height, env.im_width, 1 if settings.AGENT_IMG_TYPE == AGENT_IMAGE_TYPE.grayscaled else 3)), [0]])
#
#    # Enable preview camera with mode 0
#    env.preview_camera_enabled = settings.PREVIEW_CAMERA_RES[0]
#
#    # Main loop
#    while True:
#
#        # Carla want's agent to pause (world change) -pause and inform Carla about that
#        if pause.value == 1:
#            pause.value = 2
#
#        # Wait for Carla to release pause
#        if pause.value == 2:
#            time.sleep(0.1)
#            continue
#
#        # Pause lock released, reconnect and play
#        if pause.value == 3:
#            pause.value = 0
#            try:
#                env.destroy_agents()
#            except:
#                pass
#            try:
#                env = CarlaEnv(0, playing=True)
#                env.frametimes = deque(maxlen=60)
#                env.preview_camera_enabled = settings.PREVIEW_CAMERA_RES[0]
#            except:
#                pass
#
#            # Sleep for a second and restart episode
#            time.sleep(1)
#            continue
#
#        # Destroy agents if there are any and restart environment
#        current_state = env.reset()
#        env.collision_hist = []
#
#        # Prepare image accordingly to settings
#        current_state[0] = agent.prepare_image(current_state[0], create=True)
#
#        # For synced mode
#        last_processed_cam_update = 0
#
#        # Reset min and max values for Convcam
#        conv_min = None
#        conv_max = None
#
#        # Loop over steps
#        while True:
#
#            # For FPS measurement
#            step_start = time.time()
#
#            # If synced more - wait for updated frame, but no longer than a second
#            if settings.AGENT_SYNCED:
#                wait_for_frame_start = time.time()
#                while True:
#                    if env.last_cam_update > last_processed_cam_update:
#                        last_processed_cam_update = env.last_cam_update
#                        break
#                    if time.time() > wait_for_frame_start + 1:
#                        break
#                    time.sleep(0.001)
#
#            # Get action from predicted Q values
#            qs = agent.get_qs(current_state)
#            action = np.argmax(qs[0])
#
#            # Convcam
#            if agent.show_conv_cam:
#
#                # Calculate min and max values and add them weighted - stabilizes image flickering
#                conv_min = np.min(qs[1]) if conv_min is None else 0.8*conv_min + 0.2*np.min(qs[1])
#                conv_max = np.max(qs[1]) if conv_max is None else 0.8*conv_max + 0.2*np.max(qs[1])
#
#                # Normalize to 0..255
#                conv_preview = ((qs[1] - conv_min) * 255 / (conv_max - conv_min)).astype(np.uint8)
#
#                # Swap axes and reshape to format output image
#                conv_preview = np.moveaxis(conv_preview, 1, 2)
#                conv_preview = conv_preview.reshape((conv_preview.shape[0], conv_preview.shape[1] * conv_preview.shape[2]))
#
#                # Find where to "wrap" wide image
#                i = 1
#                while not (conv_preview.shape[1] / qs[1].shape[1]) % (i * i):
#                    i *= 2
#                i //= 2
#
#                # Wrap image
#                conv_preview_reorganized = np.zeros((conv_preview.shape[0] * i, conv_preview.shape[1] // i), dtype=np.uint8)
#                for start in range(i):
#                    conv_preview_reorganized[start * conv_preview.shape[0]:(start + 1) * conv_preview.shape[0], 0:conv_preview.shape[1] // i] = conv_preview[:, (conv_preview.shape[1] // i) * start:(conv_preview.shape[1] // i) * (start + 1)]
#
#                # Show image
#                cv2.imshow(f'Agent - Convcam', conv_preview_reorganized)
#                cv2.waitKey(1)
#
#            # Step environment
#            new_state, reward, done, _ = env.step(action)
#
#            # Show a preview (env)
#            '''
#            cv2.imshow(f'Agent - preview', new_state[0])
#            cv2.waitKey(1)
#            '''
#
#            # Prepare observation space accordingly to settings
#            new_state[0] = agent.prepare_image(new_state[0])
#
#            # Show a preview (agent)
#            '''
#            cv2.imshow(f'Agent - preview', new_state[0])
#            cv2.waitKey(1)
#            '''
#
#            # Show preview ("above the car" camera)
#            if env.preview_camera is not None:
#                cv2.imshow(f'Agent - preview', env.preview_camera)
#                cv2.waitKey(1)
#
#            # Set new state as current state (for next step)
#            current_state = new_state
#
#            # If 'done' flag from environment is set - end of an episode
#            if done or pause.value > 0:
#                break
#
#            # Frame time, also whole episode time (including sleeping, for agent FPS measurement)
#            frame_time = time.time() - step_start
#            fps_counter.append(frame_time)
#
#            console_print_callback(fps_counter, env, qs[0], action, ACTIONS_NAMES[env.actions[action]])
#
#        # Episode ended, remove actors
#        env.destroy_agents()

