import numpy as np
from scipy import stats
import tensorflow as tf


class EpisodeManager:

    def __init__(self, output_len):
        self._output_len = output_len

        self.observationList = []
        self.actionNestedList = []
        self.rewardsList = []
        self.total_reward = 0

    def add(self, observation, action_index, reward):
        actions = np.zeros(self._output_len)
        actions[action_index] = 1

        self.observationList.append(observation)
        self.actionNestedList.append(actions)
        self.rewardsList.append(reward)
        self.total_reward += reward

    def clear_all(self):
        self.observationList.clear()
        self.actionNestedList.clear()
        self.rewardsList.clear()
        self.total_reward = 0

    def get_stack_transposed(self):
        obs = np.vstack(self.observationList)
        actions = np.vstack(np.array(self.actionNestedList))
        return obs.T, actions.T


class PolicyGradientNetwork:

    # Every time we get a working version, we save it and iterate model from here.
    filepath = "./model.ckpt"

    def __init__(self, env, learning_rate = 0.02, reward_dep_rate = 0.99):
        """
        :param env: Environment state.
        :param learning_rate: Step size of gradient.
        :param gamma: Reward's depreciation rate per step.
        """

        self._has_run = False

        # Network essentials
        self._input_len = env.observation_space.shape[0]
        self._output_len = env.action_space.n
        self._learning_rate = learning_rate
        self._gamma = reward_dep_rate

        # Episodes
        self._episode_manager = EpisodeManager(self._output_len)

        # Set up Neural Network.
        layer_size, seed = 10, 33
        self._X, self._Y, self._adjusted_rewards = self._create_inputs()
        layer_vars = self._create_parameters(layer_size, seed)
        Z = self._propagate(layer_vars)
        logits = tf.transpose(Z)
        self._softmax_outputs = tf.nn.softmax(logits)
        self.training_optimizer = self._create_loss_optimizer(logits)

        # Configure TensorFlow
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self._saver = tf.train.Saver()

        # Restore model from file.
        if self._has_run:
            self._saver.restore(self._sess, PolicyGradientNetwork.filepath)

    def get_total_reward(self):
        return self._episode_manager.total_reward

    def _create_inputs(self):
        with tf.name_scope('inputs'):
            X = tf.placeholder(tf.float32, shape = (self._input_len, None), name = "_input_len")
            Y = tf.placeholder(tf.float32, shape = (self._output_len, None), name = "_output_len")
            adjusted_rewards = tf.placeholder(tf.float32, [None, ], name = "_adjusted_rewards")

            return X, Y, adjusted_rewards

    def _create_parameters(self, layer_size, seed):
        with tf.name_scope('parameters'):
            xavier_initializer = tf.contrib.layers.xavier_initializer(seed)

            # Input layer to first hidden layer.
            weight_in = tf.get_variable('weight_in', [layer_size, self._input_len],
                                      initializer = xavier_initializer)
            yint_in = tf.get_variable('yint_in', [layer_size, 1],
                                    initializer = xavier_initializer)

            # First hidden layer to second hidden layer.
            weight_hidden = tf.get_variable('weight_hidden', [layer_size, layer_size],
                                            initializer = xavier_initializer)
            yint_hidden = tf.get_variable('yint_hidden', [layer_size, 1],
                                          initializer = xavier_initializer)

            # Second hidden layer to output layer.
            weight_out = tf.get_variable('weight_out', [self._output_len, layer_size],
                                         initializer = xavier_initializer)
            yint_out = tf.get_variable('yint_out', [self._output_len, 1],
                                       initializer = xavier_initializer)

        return [(weight_in, yint_in), (weight_hidden, yint_hidden), (weight_out, yint_out)]

    def _propagate(self, layer_vars):
        Z = None
        A = self._X

        for index, pair in enumerate(layer_vars):
            with tf.name_scope('layer_{}'.format(index)):
                Z = tf.add(tf.matmul(pair[0], A), pair[1])
                A = tf.nn.relu(Z)

        return Z

    def _create_loss_optimizer(self, logits):
        labels = tf.transpose(self._Y)

        with tf.name_scope('loss'):
            prob = tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
            loss = tf.reduce_mean(prob * self._adjusted_rewards)

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)

        return optimizer

    def add_episode(self, observation, action_index, reward):
        self._episode_manager.add(observation, action_index, reward)

    def choose_action(self, observation):
        """Returns index of action to choose from action list."""
        feed_dict = {self._X: observation[:, np.newaxis]}
        weights = self._sess.run(self._softmax_outputs, feed_dict)
        indices = range(len(weights.ravel()))

        return np.random.choice(indices, p = weights.ravel())

    def learn(self):
        adjusted_rewards = stats.zscore(self._discount_rewards())
        self._train(adjusted_rewards)
        self._episode_manager.clear_all()

    def save_model(self):
        self._saver.save(self._sess, PolicyGradientNetwork.filepath)
        self._has_run = True

    def _discount_rewards(self):
        discounted_rewards = np.zeros_like(self._episode_manager.rewardsList)
        cumulative_reward = 0

        for index, reward in enumerate(reversed(self._episode_manager.rewardsList)):
            cumulative_reward *= self._gamma
            cumulative_reward += reward
            discounted_rewards[index] = cumulative_reward

        return discounted_rewards

    def _train(self, adjusted_rewards):
        obs_t, act_t = self._episode_manager.get_stack_transposed()

        feed_dict = {
            self._X: obs_t,
            self._Y: act_t,
            self._adjusted_rewards: adjusted_rewards
        }

        self._sess.run(self.training_optimizer, feed_dict)
