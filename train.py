from multi_user_network_env import env_network
from drqn import QNetwork, Memory
import numpy as np
import sys
import matplotlib.pyplot as plt 
from collections import deque
import os
import tensorflow as tf
import time

TIME_SLOTS = 100000
NUM_CHANNELS = 2
NUM_USERS = 3
ATTEMPT_PROB = 1

def one_hot(num, length):
    assert num >= 0 and num < length, "error"
    vec = np.zeros([length], np.int32)
    vec[num] = 1
    return vec

def state_generator(action, obs):
    input_vector = []
    if action is None:
        print('None')
        sys.exit()
    for user_i in range(action.size):
        input_vector_i = one_hot(action[user_i], NUM_CHANNELS + 1)
        channel_alloc = obs[-1]
        input_vector_i = np.append(input_vector_i, channel_alloc)
        input_vector_i = np.append(input_vector_i, int(obs[user_i][0]))
        input_vector.append(input_vector_i)
    return input_vector

memory_size = 1000
batch_size = 6
pretrain_length = batch_size
hidden_size = 128
learning_rate = 0.0001
explore_start = 0.02
explore_stop = 0.01
decay_rate = 0.0001
gamma = 0.9
noise = 0.1
step_size = 5
state_size = 2 * (NUM_CHANNELS + 1)
action_size = NUM_CHANNELS + 1
alpha = 0
beta = 1

# Initialize the environment
env = env_network(NUM_USERS, NUM_CHANNELS, ATTEMPT_PROB)

# Initialize the deep Q network
mainQN = QNetwork(name='main', hidden_size=hidden_size, learning_rate=learning_rate, 
                  step_size=step_size, state_size=state_size, action_size=action_size)

# Experience replay buffer
memory = Memory(max_size=memory_size)

# Input buffer for predicting next Q-values
history_input = deque(maxlen=step_size)

# Sample random actions for each user
action = env.sample()
obs = env.step(action)
state = state_generator(action, obs)
reward = [i[1] for i in obs[:NUM_USERS]]

# Pre-train the memory buffer
for ii in range(pretrain_length * step_size * 5):
    action = env.sample()
    obs = env.step(action)
    next_state = state_generator(action, obs)
    reward = [i[1] for i in obs[:NUM_USERS]]
    memory.add((state, action, reward, next_state))
    state = next_state
    history_input.append(state)

##############################################
    
def get_states(batch): 
    states = []
    for  i in batch:
        states_per_batch = []
        for step_i in i:
            states_per_step = []
            for user_i in step_i[0]:
                states_per_step.append(user_i)
            states_per_batch.append(states_per_step)
        states.append(states_per_batch)     
   
    return states

def get_actions(batch):
    actions = []
    for each in batch:
        actions_per_batch = []
        for step_i in each:
            actions_per_step = []
            for user_i in step_i[1]:
                actions_per_step.append(user_i)
            actions_per_batch.append(actions_per_step)
        actions.append(actions_per_batch)

    return actions

def get_rewards(batch):
    rewards = []
    for each in batch:
        rewards_per_batch = []
        for step_i in each:
            rewards_per_step = []
            for user_i in step_i[2]:
                rewards_per_step.append(user_i)
            rewards_per_batch.append(rewards_per_step)
        rewards.append(rewards_per_batch)
    return rewards

def get_next_states(batch):
    next_states = []
    for each in batch:
        next_states_per_batch = []
        for step_i in each:
            next_states_per_step = []
            for user_i in step_i[3]:
                next_states_per_step.append(user_i)
            next_states_per_batch.append(next_states_per_step)
        next_states.append(next_states_per_batch)
    return next_states        

def get_states_user(batch):
    states = []
    for user in range(NUM_USERS):
        states_per_user = []
        for each in batch:
            states_per_batch = []
            for step_i in each:
                
                try:
                    states_per_step = step_i[0][user]
                    
                except IndexError:
                    print (step_i)
                    print ("-----------")
                    
                    print ("eror")
                    
                    '''for i in batch:
                        print i
                        print "**********"'''
                    sys.exit()
                states_per_batch.append(states_per_step)
            states_per_user.append(states_per_batch)
        states.append(states_per_user)
    #print len(states)
    return np.array(states)

def get_actions_user(batch):
    actions = []
    for user in range(NUM_USERS):
        actions_per_user = []
        for each in batch:
            actions_per_batch = []
            for step_i in each:
                actions_per_step = step_i[1][user]
                actions_per_batch.append(actions_per_step)
            actions_per_user.append(actions_per_batch)
        actions.append(actions_per_user)
    return np.array(actions)

def get_rewards_user(batch):
    rewards = []
    for user in range(NUM_USERS):
        rewards_per_user = []
        for each in batch:
            rewards_per_batch = []
            for step_i in each:
                rewards_per_step = step_i[2][user] 
                rewards_per_batch.append(rewards_per_step)
            rewards_per_user.append(rewards_per_batch)
        rewards.append(rewards_per_user)
    return np.array(rewards)




# 
def get_next_states_user(batch):
    next_states = []
    for user in range(NUM_USERS):
        next_states_per_user = []
        for each in batch:
            next_states_per_batch = []
            for step_i in each:
                next_states_per_step = step_i[3][user] 
                next_states_per_batch.append(next_states_per_step)
            next_states_per_user.append(next_states_per_batch)
        next_states.append(next_states_per_user)
    return np.array(next_states)



interval = 1

# Initialize the model checkpointing
checkpoint_dir = './checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=mainQN.opt, model=mainQN)

# Initialize the cumulative reward and collision
total_rewards = []
cum_r = [0]
cum_collision = [0]

for time_step in range(TIME_SLOTS):
    if time_step % 50 == 0:
        if time_step < 5000:
            beta -= 0.001

    explore_p = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * time_step)

    if explore_p > np.random.rand():
        action = env.sample()
        print("explored")
    else:
        action = np.zeros([NUM_USERS], dtype=np.int32)
        state_vector = np.array(history_input)
        for each_user in range(NUM_USERS):
            Qs = mainQN(state_vector[:, each_user].reshape(1, step_size, state_size))
            prob1 = (1 - alpha) * np.exp(beta * Qs)
            prob = prob1 / np.sum(np.exp(beta * Qs)) + alpha / (NUM_CHANNELS + 1)
            action[each_user] = np.argmax(prob, axis=1)

    obs = env.step(action)
    next_state = state_generator(action, obs)
    reward = [i[1] for i in obs[:NUM_USERS]]
    sum_r = np.sum(reward)
    cum_r.append(cum_r[-1] + sum_r)
    collision = NUM_CHANNELS - sum_r
    cum_collision.append(cum_collision[-1] + collision)

    for i in range(len(reward)):
        if reward[i] > 0:
            reward[i] = sum_r

    total_rewards.append(sum_r)
    memory.add((state, action, reward, next_state))
    state = next_state
    history_input.append(state)

    batch = memory.sample(batch_size, step_size)
    states = get_states_user(batch)
    actions = get_actions_user(batch)
    rewards = get_rewards_user(batch)
    next_states = get_next_states_user(batch)

    states = np.reshape(states, [-1, states.shape[2], states.shape[3]])
    actions = np.reshape(actions, [-1, actions.shape[2]])
    rewards = np.reshape(rewards, [-1, rewards.shape[2]])
    next_states = np.reshape(next_states, [-1, next_states.shape[2], next_states.shape[3]])

    target_Qs = mainQN(next_states)
    targets = rewards[:, -1] + gamma * np.max(target_Qs, axis=1)

    with tf.GradientTape() as tape:
        # Get the Q-values for the current states
        Q = mainQN(states)  # Q has shape [batch_size, action_size]
        
        # Select the Q-values corresponding to the actions taken
        actions_taken = actions[:, -1]  # Get the actions taken in the last time step
        Q_taken = tf.reduce_sum(Q * tf.one_hot(actions_taken, action_size), axis=1)
        
        # Compute the loss between the selected Q-values and the targets
        loss = tf.reduce_mean(tf.square(targets - Q_taken))


    gradients = tape.gradient(loss, mainQN.trainable_variables)
    mainQN.opt.apply_gradients(zip(gradients, mainQN.trainable_variables))

    if time_step % 5000 == 4999:
        plt.figure(1)
        plt.subplot(211)
        plt.plot(np.arange(5001), cum_collision, "r-")
        plt.xlabel('Time Slot')
        plt.ylabel('Cumulative collision')
        plt.subplot(212)
        plt.plot(np.arange(5001), cum_r, "r-")
        plt.xlabel('Time Slot')
        plt.ylabel('Cumulative reward of all users')
        plt.show()

        total_rewards = []
        cum_r = [0]
        cum_collision = [0]
        checkpoint.save(file_prefix=checkpoint_prefix)
        print(time_step, loss.numpy(), sum(reward))

    print("*************************************************")
