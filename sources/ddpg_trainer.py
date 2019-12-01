import os
import sys
import settings
from sources import ARTDDPGAgent, TensorBoard, STOP, ACTIONS, ACTIONS_NAMES
from collections import deque
import time
import random
import numpy as np
import pickle
import json
from dataclasses import dataclass
from threading import Thread

# Try to mute and then load Tensorflow
# Muting seems to not work lately on Linux in any way
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#stdin = sys.stdin
#sys.stdin = open(os.devnull, 'w')
#stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
import keras.backend.tensorflow_backend as backend
#sys.stdin = stdin
#sys.stderr = stderr


# Trainer class
class ARTDDPGTrainer(ARTDDPGAgent):
    def __init__(self, model_path):

        # If model path is beiong passed in - use it instead of creating a new one
        self.model_path = model_path

        # Main model (agent does not use target model)
        self.actor = self.create_model(is_actor=True)
        self.critic = self.create_model(is_actor=False)
        self.action_grads = backend.function(self.critic.inputs,
                                             backend.gradients(self.critic.output, [self.critic.inputs[1]]))

        action_gdts = backend.placeholder(shape=(None, len(settings.ACTIONS)))
        params_grad = tf.gradients(self.actor.output, self.actor.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.actor.trainable_weights)
        self.actor_optimizer = backend.function([self.actor.input, action_gdts],
                                                [tf.train.AdamOptimizer(self.get_lr_decay()[0]).apply_gradients(grads)])

        self.tau = settings.TAU

        # We are going to train a model in a loop using separate thread
        # Tensorflow needs to know about the graph to use as we load or create model in main thread
        # Save model graph as object property for later use
        self.graph = tf.get_default_graph()

    # Init is being split into two parts. Firs one is being loded always, but this one only for training
    # (when calculating weights we don't need that)
    def init2(self, stop, logdir, trainer_stats, episode, epsilon, discount, update_target_every, last_target_update, min_reward, agent_show_preview, save_checkpoint_every, seconds_per_episode, duration, optimizer, models, car_npcs):

        # Trainer does not use convcam
        self.show_conv_cam = False

        # Target networks
        self.target_critic = self.create_model(prediction=True, is_actor=False)
        self.target_critic.set_weights(self.critic.get_weights())

        self.target_actor = self.create_model(prediction=True, is_actor=True)
        self.target_actor.set_weights(self.actor.get_weights())

        # An array with last n transitions for training
        self.replay_memory = deque(maxlen=settings.REPLAY_MEMORY_SIZE)

        # Set log dir for tensorboard - either create one or use (if passed in) existing one
        # Create tensorboard object and set current step (being an episode for the agent)
        self.logdir = logdir if logdir else "logs/{}-{}-{}".format(settings.ALGORITHM, settings.MODEL_NAME,
                                                                   int(time.time()))
        self.tensorboard = TensorBoard(log_dir=self.logdir)
        self.tensorboard.step = episode.value

        # Used to count when to update target network with main network's weights
        self.last_target_update = last_target_update

        # Internal properties
        self.last_log_episode = 0
        self.tps = 0
        self.last_checkpoint = 0
        self.save_model = False

        # Shared properties - either used by model or only for checkpoint purposes
        self.stop = stop
        self.trainer_stats = trainer_stats
        self.episode = episode
        self.epsilon = epsilon
        self.discount = discount
        self.update_target_every = update_target_every
        self.min_reward = min_reward
        self.agent_show_preview = agent_show_preview
        self.save_checkpoint_every = save_checkpoint_every
        self.seconds_per_episode = seconds_per_episode
        self.duration = duration
        self.optimizer = optimizer
        self.models = models
        self.car_npcs = car_npcs

        # Update optimizer stats with lr and decay
        self.optimizer[0], self.optimizer[1] = self.get_lr_decay()

    # Adds transition (step's data) to a memory replay list
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self):

        # Start training only if certain number of transitions is already being saved in replay memory
        if len(self.replay_memory) < settings.MIN_REPLAY_MEMORY_SIZE:
            return False

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, settings.MINIBATCH_SIZE)

       # # Get current states from minibatch, then query NN model for Q values
       # current_states = [np.array([transition[0][0] for transition in minibatch])/255]

       # # We need to use previously saved graph here as this is going to be called from separate thread
       # with self.graph.as_default():
       #     current_qs_list = self.model.predict(current_states, settings.PREDICTION_BATCH_SIZE)

        # Get future states from minibatch, then query target critic model for actions
        # When using target network, query it, otherwise main network should be queried
        new_current_states = [np.array([transition[3][0] for transition in minibatch])/255]

        with self.graph.as_default():
            future_actions_list = self.target_actor.predict(new_current_states, settings.PREDICTION_BATCH_SIZE)

        with self.graph.as_default():
            future_qs_list = self.target_critic.predict([new_current_states, future_actions_list],
                                                        settings.PREDICTION_BATCH_SIZE)

        states_normalized_input = []
        actions_input = []
        critic_target = []

        # Enumerate samples in minibatch
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If it's not a terminal state, get new Q value from future states, otherwise set it to a reward
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                new_q = reward + self.discount.value * future_qs_list[index]
            else:
                new_q = reward

            ## Update Q value for given state
            #current_qs = current_qs_list[index]
            #current_qs[action] = new_q

            # And append to our training data
            states_normalized_input.append(np.array(current_state[0])/255)
            actions_input.append(action)
            critic_target.append(new_q)

        # Log only on terminal state. As trainer trains in an asynchronous way, it does not know when
        # and which agent just finished an episode. Instead of that we monitor episode number and once
        # it changes, we log current .fit() call. We do that as we do want to save stats once per every episode
        log_this_step = False
        if self.tensorboard.step > self.last_log_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        # Fit on all samples as one batch
        with self.graph.as_default():
            self.critic.fit(np.array([states_normalized_input, actions_input]), np.array(critic_target),
                            batch_size=settings.TRAINING_BATCH_SIZE, verbose=0, shuffle=False,
                            callbacks=[self.tensorboard] if log_this_step else None)

            # Update optimizer with new values if there are nay
            if self.optimizer[2] == 1:
                self.optimizer[2] = 0
                #backend.set_value(self.model.optimizer.lr, self.optimizer[3])
                self.compile_model(model=self.model, lr=self.optimizer[3], decay=self.get_lr_decay()[1])
            if self.optimizer[4] == 1:
                self.optimizer[4] = 0
                #backend.set_value(self.model.optimizer.decay, self.optimizer[5])
                self.compile_model(model=self.model, lr=self.get_lr_decay()[0], decay=self.optimizer[5])

            # Update optimizer stats with lr and decay
            self.optimizer[0], self.optimizer[1] = self.get_lr_decay()

        current_states = [np.array([transition[0][0] for transition in minibatch])/255]

        with self.graph.as_default():
            current_actions_list = self.actor.predict(current_states, settings.PREDICTION_BATCH_SIZE)
            gradients = self.action_grads([current_states, actions_input])
            self.actor_optimizer([current_actions_list, gradients])


        # If step counter reaches set value, update target network with weights of main network
        if self.tensorboard.step >= self.last_target_update + self.update_target_every.value:
            W, target_W = self.actor.get_weights(), self.target_actor.get_weights()
            for i in range(W):
                target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
            self.target_actor.set_weights(target_W)

            W, target_W = self.critic.get_weights(), self.target_critic.get_weights()
            for i in range(W):
                target_W[i] = self.tau * W[i] + (1 - self.tau) * target_W[i]
            self.target_critic.set_weights(target_W)

            self.last_target_update += self.update_target_every.value

        return True

    # Returns current learning rate and decay values from Adam optimizer
    def get_lr_decay(self):
        lr = self.critic.optimizer.lr
        if self.critic.optimizer.initial_decay > 0:
            lr = lr * (1. / (1. + self.critic.optimizer.decay * backend.cast(self.critic.optimizer.iterations, backend.dtype(self.critic.optimizer.decay))))
        return backend.eval(lr), backend.eval(self.critic.optimizer.decay)

    # Prepares weights to be send to agents over shared object
    def serialize_weights(self):
        weights = {'actor': pickle.dumps(self.actor.get_weights()),
                   'critic': pickle.dumps(self.critic.get_weights())}
        return weights

    # Creates first set of weights to agents to load when they start
    # Uses shared object, updates it and updates weights iteration counter so agents can see a change
    def init_serialized_weights(self, weights_actor, weights_critic, weights_iteration):
        self.weights_actor = weights_actor
        self.weights_critic = weights_critic
        weights_raw = self.serialize_weights()
        self.weights_critic.raw = weights_raw['critic']
        self.weights_actor.raw = weights_raw['actor']
        self.weights_iteration = weights_iteration

    # Trains model in a loop, calles from a separate thread
    def train_in_loop(self):
        self.tps_counter = deque(maxlen=20)

        # Train infinitively
        while True:

            # For training speed measurement
            step_start = time.time()

            # If 'stop' flag - exit
            if self.stop.value == STOP.stopping:
                return

            # If Carla broke - pause training
            if self.stop.value in [STOP.carla_simulator_error, STOP.restarting_carla_simulator]:
                self.trainer_stats[0] = TRAINER_STATE.paused
                time.sleep(1)
                continue

            # If .train() call returns false, there's not enough transitions in replay memory
            # Just wait (and exit on 'stop' signal)
            if not self.train():
                self.trainer_stats[0] = TRAINER_STATE.waiting

                # Trainer is also a manager for stopping everything as it has to save a checkpoint
                if self.stop.value in [STOP.at_checkpoint, STOP.now]:
                    self.stop.value = STOP.stopping

                time.sleep(0.01)
                continue

            # If we are here, trainer trains a model
            self.trainer_stats[0] = TRAINER_STATE.training

            # Share new weights with models as fast as possible
            weights_raw = self.serialize_weights()
            self.weights_critic.raw = weights_raw['critic']
            self.weights_actor.raw = weights_raw['actor']
            with self.weights_iteration.get_lock():
                self.weights_iteration.value += 1

            # Training part finished here, measure time and convert it to number of trains per second
            frame_time = time.time() - step_start
            self.tps_counter.append(frame_time)
            self.trainer_stats[1] = len(self.tps_counter)/sum(self.tps_counter)

            # Shared flag set by models when they performed good to save a model
            save_model = self.save_model
            if save_model:
                self.actor.save(save_model)
                self.critic.save(save_model)
                self.save_model = False

            # Checkpoint - if given number of episodes passed, save a checkpoint
            # Checkpoints does not contain all data, they do not include stats,
            # but stats are not important for training. Checkpoint does not contain replay memory
            # as saving several GB of data wil slow things down significantly and
            # fill-up disk space quickly
            checkpoint_number = self.episode.value // self.save_checkpoint_every.value

            # Save every nth step and on 'stop' flag
            if checkpoint_number > self.last_checkpoint or self.stop.value == STOP.now:

                # Create and save hparams file
                self.models.append(f'checkpoint/{settings.MODEL_NAME}_{self.episode.value}.model')
                hparams = {
                    'duration': self.duration.value,
                    'episode': self.episode.value,
                    'epsilon': list(self.epsilon),
                    'discount': self.discount.value,
                    'update_target_every': self.update_target_every.value,
                    'last_target_update': self.last_target_update,
                    'min_reward': self.min_reward.value,
                    'agent_show_preview': [list(preview) for preview in self.agent_show_preview],
                    'save_checkpoint_every': self.save_checkpoint_every.value,
                    'seconds_per_episode': self.seconds_per_episode.value,
                    'model_path': f'checkpoint/{settings.MODEL_NAME}_{self.episode.value}.model',
                    'logdir': self.logdir,
                    'weights_iteration': self.weights_iteration.value,
                    'car_npcs': list(self.car_npcs),
                    'models': list(set(self.models))
                }

                # Save the model
                self.critic.save(f'checkpoint/critic_{settings.MODEL_NAME}_{hparams["episode"]}.model')
                self.actor.save(f'checkpoint/actor_{settings.MODEL_NAME}_{hparams["episode"]}.model')

                with open('checkpoint/hparams_new.json', 'w', encoding='utf-8') as f:
                    json.dump(hparams, f)

                try:
                    os.remove('checkpoint/hparams.json')
                except:
                    pass
                try:
                    os.rename('checkpoint/hparams_new.json', 'checkpoint/hparams.json')
                    self.last_checkpoint = checkpoint_number
                except Exception as e:
                    print(str(e))

            # Handle for 'stop' signal
            if self.stop.value in [STOP.at_checkpoint, STOP.now]:
                self.stop.value = STOP.stopping


# Trainer states
@dataclass
class TRAINER_STATE:
    starting = 0
    waiting = 1
    training = 2
    finished = 3
    paused = 4


# Trainer state messages
TRAINER_STATE_MESSAGE = {
    0: 'STARTING',
    1: 'WAITING',
    2: 'TRAINING',
    3: 'FINISHED',
    4: 'PAUSED',
}


# Creates a model, dumps weights and saves this number
# We need this side to know how big shared object to create
def check_weights_size(model_path, weights_size_actor, weights_size_critic):

    # Memory fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings.TRAINER_MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # create a model and save serialized weights' size
    trainer = ARTDDPGTrainer(model_path)
    weights = trainer.serialize_weights()
    weights_size_actor.value = len(weights['actor'])
    weights_size_critic.value = len(weights['critic'])


# Runs trainer process
def run(model_path, logdir, stop, weights_actor, weights_critic, weights_iteration, episode, epsilon, discount, update_target_every, last_target_update, min_reward, agent_show_preview, save_checkpoint_every, seconds_per_episode, duration, transitions, tensorboard_stats, trainer_stats, episode_stats, optimizer, models, car_npcs, carla_settings_stats, carla_fps):

    # Set GPU used for the trainer
    if settings.TRAINER_GPU is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]= str(settings.TRAINER_GPU)

    tf.set_random_seed(1)
    random.seed(1)
    np.random.seed(1)

    # Memory fraction
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=settings.TRAINER_MEMORY_FRACTION)
    backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

    # Create trainer, run second init method and initialize weights so agents can load them
    trainer = ARTDDPGTrainer(model_path)
    trainer.init2(stop, logdir, trainer_stats, episode, epsilon, discount, update_target_every, last_target_update, min_reward, agent_show_preview, save_checkpoint_every, seconds_per_episode, duration, optimizer, models, car_npcs)
    trainer.init_serialized_weights(weights_actor, weights_critic, weights_iteration)

    trainer_stats[0] = TRAINER_STATE.waiting

    # Create training thread. We train in a separate thread so training won't block other things we do here
    trainer_thread = Thread(target=trainer.train_in_loop, daemon=True)
    trainer_thread.start()

    # Helper deques for stat averaging
    raw_rewards = deque(maxlen=settings.AGENTS*10)
    weighted_rewards = deque(maxlen=settings.AGENTS*10)
    episode_times = deque(maxlen=settings.AGENTS*10)
    frame_times = deque(maxlen=settings.AGENTS*2)

    configured_actions = [getattr(ACTIONS, action) for action in settings.ACTIONS]

    # Iterate over episodes until 'stop' signal
    while stop.value != 3:

        # Update tensorboard step every episode
        if episode.value > trainer.tensorboard.step:
            trainer.tensorboard.step = episode.value

        # Load new transitions put here by models and place then im memory replay table
        for _ in range(transitions.qsize()):
            try:
                trainer.update_replay_memory(transitions.get(True, 0.1))
            except:
                break

        # Log stats in tensorboard
        while not tensorboard_stats.empty():

            # Added to a Queue by agents
            agent_episode, reward, agent_epsilon, episode_time, frame_time, weighted_reward, *avg_predicted_qs = tensorboard_stats.get_nowait()

            # Append to lists for averaging
            raw_rewards.append(reward)
            weighted_rewards.append(weighted_reward)
            episode_times.append(episode_time)
            frame_times.append(frame_time)

            # All monitored stats
            episode_stats[0] = min(raw_rewards)  # Minimum reward (raw)
            episode_stats[1] = sum(raw_rewards)/len(raw_rewards)  # Average reward (raw)
            episode_stats[2] = max(raw_rewards)  # Maximum reward (raw)
            episode_stats[3] = min(episode_times)  # Minimum episode duration
            episode_stats[4] = sum(episode_times)/len(episode_times)  # Average episode duration
            episode_stats[5] = max(episode_times)  # Maximum episode duration
            episode_stats[6] = sum(frame_times)/len(frame_times)  # Average agent FPS
            episode_stats[7] = min(weighted_rewards)  # Minimum reward (weighted)
            episode_stats[8] = sum(weighted_rewards)/len(weighted_rewards)  # Average reward (weighted)
            episode_stats[9] = max(weighted_rewards)  # Maximum reward (weighted)
            tensorboard_q_stats = {}
            for action, (avg_predicted_q, std_predicted_q, usage_predicted_q) in enumerate(zip(avg_predicted_qs[0::3], avg_predicted_qs[1::3], avg_predicted_qs[2::3])):
                if avg_predicted_q != -10**6:
                    episode_stats[action*3 + 10] = avg_predicted_q
                    tensorboard_q_stats[f'q_action_{action - 1}_{ACTIONS_NAMES[configured_actions[action - 1]]}_avg' if action else f'q_all_actions_avg'] = avg_predicted_q
                if std_predicted_q != -10 ** 6:
                    episode_stats[action*3 + 11] = std_predicted_q
                    tensorboard_q_stats[f'q_action_{action - 1}_{ACTIONS_NAMES[configured_actions[action - 1]]}_std' if action else f'q_all_actions_std'] = std_predicted_q
                if usage_predicted_q != -10 ** 6:
                    episode_stats[action*3 + 12] = usage_predicted_q
                    if action > 0:
                        tensorboard_q_stats[f'q_action_{action - 1}_{ACTIONS_NAMES[configured_actions[action - 1]]}_usage_pct'] = usage_predicted_q
            carla_stats = {}
            for process_no in range(settings.CARLA_HOSTS_NO):
                for index, stat in enumerate(['carla_{}_car_npcs', 'carla_{}_weather_sun_azimuth', 'carla_{}_weather_sun_altitude', 'carla_{}_weather_clouds_pct', 'carla_{}_weather_wind_pct', 'carla_{}_weather_rain_pct']):
                    if carla_settings_stats[process_no][index] != -1:
                        carla_stats[stat.format(process_no+1)] = carla_settings_stats[process_no][index]
                carla_stats[f'carla_{process_no + 1}_fps'] = carla_fps[process_no].value

            # Save logs
            trainer.tensorboard.update_stats(step=agent_episode, reward_raw_avg=episode_stats[1], reward_raw_min=episode_stats[0], reward_raw_max=episode_stats[2], reward_weighted_avg=episode_stats[8], reward_weighted_min=episode_stats[7], reward_weighted_max=episode_stats[9], epsilon=agent_epsilon, episode_time_avg=episode_stats[4], episode_time_min=episode_stats[3], episode_time_max=episode_stats[5], agent_fps_avg=episode_stats[6], optimizer_lr=optimizer[0], optimizer_decay=optimizer[1], **tensorboard_q_stats, **carla_stats)

            # Save model, but only when min reward is greater or equal a set value
            if episode_stats[7] >= min_reward.value:
                trainer.save_model = f'models/{settings.MODEL_NAME}__{episode_stats[2]:_>7.2f}max_{episode_stats[1]:_>7.2f}avg_{episode_stats[0]:_>7.2f}min__{int(time.time())}.model'

        time.sleep(0.01)

    # End of training, wait for trainer thread to finish
    trainer_thread.join()

    trainer_stats[0] = TRAINER_STATE.finished
