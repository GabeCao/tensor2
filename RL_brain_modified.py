import numpy as np
import tensorflow as tf
import random
import os
from sklearn.preprocessing import LabelBinarizer

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions=1,
            n_features=117,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

        # ################从这开始都是新添加的属性###############################
        # 对action 独热编码,用字典表示,key: action(string), value: action的编码结果(ndarray)
        self.action_one_hot_encoded_8 = {}
        self.action_one_hot_encoded_9 = {}
        self.action_one_hot_encoded_10 = {}
        self.action_one_hot_encoded_11 = {}
        self.action_one_hot_encoded_12 = {}
        self.action_one_hot_encoded_13 = {}
        self.action_one_hot_encoded_14 = {}
        self.action_one_hot_encoded_15 = {}
        self.action_one_hot_encoded_16 = {}
        self.action_one_hot_encoded_17 = {}
        self.action_one_hot_encoded_18 = {}
        self.action_one_hot_encoded_19 = {}
        self.action_one_hot_encoded_20 = {}
        self.action_one_hot_encoded_21 = {}
        self.set_action_one_hot_encoded()

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    # 从这里开始都是新添加的函数
    # 传入一个时间段，返回该时间段的对action 独热编码
    def get_current_action_one_hot_encoded(self, hour):
        if hour == 8:
            return self.action_one_hot_encoded_8
        elif hour == 9:
            return self.action_one_hot_encoded_9
        elif hour == 10:
            return self.action_one_hot_encoded_10
        elif hour == 11:
            return self.action_one_hot_encoded_11
        elif hour == 12:
            return self.action_one_hot_encoded_12
        elif hour == 13:
            return self.action_one_hot_encoded_13
        elif hour == 14:
            return self.action_one_hot_encoded_14
        elif hour == 15:
            return self.action_one_hot_encoded_15
        elif hour == 16:
            return self.action_one_hot_encoded_16
        elif hour == 17:
            return self.action_one_hot_encoded_17
        elif hour == 18:
            return self.action_one_hot_encoded_18
        elif hour == 19:
            return self.action_one_hot_encoded_19
        elif hour == 20:
            return self.action_one_hot_encoded_20
        elif hour == 21:
            return self.action_one_hot_encoded_21

    ###########################################################
    # 新添加了一个传入参数，当前环境的 秒数
    def choose_action(self, observation, seconds):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            # actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            # action = np.argmax(actions_value)

            ###########################################################
            # 获得当前的时间段，通过当前时间段获得 对应的action 独热编码
            hour = int(seconds / 3600)
            current_phase = self.get_current_action_one_hot_encoded(hour)
            chose_value = 0
            chose_action = None
            # 遍历所有action 找到神经网络返回最大的action_value 的 action
            for key, value in current_phase.items():
                value = value[np.newaxis, :]
                action_observation = np.c_[value, observation]
                action_value = self.sess.run(self.q_eval, feed_dict={self.s: action_observation})
                if action_value > chose_value:
                    chose_value = action_value
                    chose_action = key
            action = chose_action
            ###########################################################

        else:
            # action = np.random.randint(0, self.n_actions)
            ###########################################################
            current_phase = self.get_current_action_one_hot_encoded
            # 随机从current_phase 选出一个action，代码其他的地方好像用的是伪随机数
            action, _ = random.choice(list(current_phase.items()))
            ###########################################################
        return action

    # 在八点时刻，随机选一个action。这个函数供环境在初始化的时候使用
    def random_action(self):
        action, _ = random.choice(list(self.action_one_hot_encoded_8.items()))
        return action

    def action_to_one_hot_encoded(self, path, file, action_one_hot_encoded):
        # 存放每个文件中的所有hotspot
        hotspots = []
        # 存放每个hotspot 对应的最大等待时间
        hotspot_max_staying_time = {}
        with open(path + file) as f:
            for line in f:
                data = line.strip().split(',')
                hotspots.append(data[0])
                hotspot_max_staying_time[data[0]] = data[1]
        # 对hotpsots 进行独热编码
        label_binarizer = LabelBinarizer()
        hotspot_one_hot_encoded = label_binarizer.fit_transform(hotspots)
        # 获取独热编码后的行和列
        rows, cols = hotspot_one_hot_encoded.shape

        for row in range(rows):
            # 获得独热编码前的hotspot 编号，用这个编号在hotspot_max_staying_time 找到其最大等待时间
            hotspot = label_binarizer.inverse_transform(hotspot_one_hot_encoded[[row]])[0]
            for key, value in hotspot_max_staying_time.items():
                if hotspot == key:
                    max_staying_time = int(hotspot_max_staying_time[key])
                    # 找到最大等待时间以后，对独热编码后的当前行后面添加一个等待时间(从1到最大等待时间)
                    for i in range(1, max_staying_time + 1):
                        tmp = hotspot_one_hot_encoded[row]
                        tmp = np.r_[tmp, i]
                        action_one_hot_encoded[hotspot + ',' + str(i)] = tmp

    def set_action_one_hot_encoded(self):
        # 保存最大等待时间的文件夹
        path = '五分钟/'
        files = os.listdir(path)
        # 对于每个文件,读取文件的内容, 进行热编码放入相应的字典中。例如8.txt代表8时间段的内容，
        # 放入到action_one_hot_encoded_8字典中
        for file in files:
            if file == '8.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_8)
            elif file == '9.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_9)
            elif file == '10.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_10)
            elif file == '11.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_11)
            elif file == '12.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_12)
            elif file == '13.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_13)
            elif file == '14.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_14)
            elif file == '15.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_15)
            elif file == '16.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_16)
            elif file == '17.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_17)
            elif file == '18.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_18)
            elif file == '19.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_19)
            elif file == '20.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_20)
            elif file == '21.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_21)

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()



