from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import argparse
import random
import time
import sys
import os
import cv2
import matplotlib.pyplot as plt
from vizdoom import *
import numpy as np
import tensorflow as tf
import matplotlib.ticker as ticker

from matplotlib.ticker import MaxNLocator

import skimage.color, skimage.transform

from vizdoom import *
np.set_printoptions(threshold=np.inf)

from pydub import AudioSegment
from playsound import playsound

from pydub.playback import play
import vizdoom as vzd

from GlobalVariables_Audio import GlobalVariables_Audio

mean_scores=[]
parameter=GlobalVariables_Audio

np.set_printoptions(threshold=np.inf)
#mean_scores=[]

def MakeDir(path):
    try:
        os.makedirs(path)
    except:
        pass

Load_Model = False
Train_Model = True

test_display = False
test_write_video = False

Working_Directory = "./"
scenario_file = Working_Directory + "find.wad"

from Environment_Audio import Environment_Audio

resolution = (1,100) + (parameter.channels_audio,)
Feature='Raw_Samples'

model_path= Working_Directory+'Model_'+Feature+ '_'+ str(parameter.how_many_times) + '_Framerepeat_'+str(parameter.frame_repeat)+'/'

MakeDir(model_path)
model_name = model_path + "model"

#DEFAULT_MODEL_SAVEFILE = model_path + '/model'

def Preprocess(img):
     img = skimage.transform.resize(img, resolution)
     return img

def Display_Training(iteration, how_many_times, train_scores):
    mean_training_scores = 0
    std_training_scores = 0
    min_training_scores = 0
    max_training_scores = 0
    if (len(train_scores) > 0):
        train_scores = np.array(train_scores)
        mean_training_scores = train_scores.mean()
        std_training_scores = train_scores.std()
        min_training_scores = train_scores.min()
        max_training_scores = train_scores.max()
    print("Steps: {}/{} Episodes: {} Rewards: mean: {:.2f}, std: {:.2f}, min: {:.2f}, max: {:.2f}"
        .format(iteration, how_many_times, len(train_scores), mean_training_scores, std_training_scores,
         min_training_scores, max_training_scores),file=sys.stderr)
    mean_training_scores=round(mean_training_scores,2)
    mean_scores.append(mean_training_scores)
    #print("Mean Scores",mean_scores)
class ReplayMemory(object):
    def __init__(self, capacity):

        self.s1 = np.zeros((capacity,) + resolution, dtype=np.float64)
        self.s2 = np.zeros((capacity,) + resolution, dtype=np.float64)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def Add(self, s1, action, s2, isterminal, reward):

        self.s1[self.pos, ...] = s1
        self.a[self.pos] = action
        self.isterminal[self.pos] = isterminal
        if not isterminal:
                self.s2[self.pos, ...] = s2
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def Get(self, sample_size):

        i = random.sample(range(0, self.size-2), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]

class Model(object):
    def __init__(self, session, actions_count):

        self.session = session
        print('Training Using',Feature)

        # Create the input.
        self.s1_ = tf.placeholder(shape=[None] + list(resolution), dtype=tf.float32)
        self.q_ = tf.placeholder(shape=[None, actions_count], dtype=tf.float32)

        # Create the network.
        conv1 = tf.contrib.layers.conv2d(self.s1_, num_outputs=16, kernel_size=[3, 3], stride=[2, 2])
        conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=32, kernel_size=[3, 3], stride=[2, 2])
        conv2_flat = tf.contrib.layers.flatten(conv2)
        fc1 = tf.contrib.layers.fully_connected(conv2_flat, num_outputs=128)
        self.q = tf.contrib.layers.fully_connected(fc1, num_outputs=actions_count, activation_fn=None)

        self.action = tf.argmax(self.q, 1)
        self.loss = tf.losses.mean_squared_error(self.q_, self.q)
        self.optimizer = tf.train.RMSPropOptimizer(parameter.Learning_Rate)
        self.train_step = self.optimizer.minimize(self.loss)

    def Learn(self, state, q):

        state = state.astype(np.float32)
        l, _ = self.session.run([self.loss, self.train_step], feed_dict={self.s1_ : state, self.q_: q})
        return l

    def GetQ(self, state):

        state = state.astype(np.float32)
        return self.session.run(self.q, feed_dict={self.s1_ : state})

    def GetAction(self, state):

        #state = state.astype(np.float32) #(30,45,3)
        state = state.reshape([1] + list(resolution))#(1, 1, 100, 1)
        return self.session.run(self.action, feed_dict={self.s1_: state})[0]

class Agent(object):
    def __init__(self, num_actions):
		
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        self.model = Model(self.session, num_actions)
        self.memory = ReplayMemory(parameter.replay_memory_size)
        #self.rewards = 0

        #self.saver = tf.train.Saver(max_to_keep=1000)
        self.saver = tf.train.Saver()
        if (Load_Model):
            print("Loading model from: ")#, DEFAULT_MODEL_SAVEFILE)
            #self.saver.restore(self.session, DEFAULT_MODEL_SAVEFILE)
            #model_name_curr = model_name #+ "_{:04}".format(step_load)
            #print("Loading model from: ", model_name_curr)
            #self.saver.restore(self.session, model_name_curr)
        else:
            init = tf.global_variables_initializer()
            self.session.run(init)

        self.num_actions = num_actions

    def LearnFromMemory(self):

        if (self.memory.size > 2*parameter.replay_memory_batch_size):
        #if (self.memory.size > parameter.replay_memory_batch_size):
            s1, a, s2, isterminal, r = self.memory.Get(parameter.replay_memory_batch_size)
            q = self.model.GetQ(s1)
            q2 = np.max(self.model.GetQ(s2), axis=1)
            q[np.arange(q.shape[0]), a] = r + (1 - isterminal) * parameter.Discount_Factor * q2
            self.model.Learn(s1, q)

    def GetAction(self, state,current_model_name):
        #print("Loading model from: ", current_model_name)
        self.saver.restore(self.session, current_model_name)
        if (random.random() <= 0.05):
            best_action = random.randint(0, self.num_actions-1)
        else:
            best_action = self.model.GetAction(state)
        return best_action

    def perform_learning_step(self, iteration):

        s1=Preprocess(env.Observation())
        # Epsilon-greedy.
        if (iteration < parameter.eps_decay_iter):
            eps = parameter.start_eps - iteration / parameter.eps_decay_iter * (parameter.start_eps - parameter.end_eps)
        else:
            eps = parameter.end_eps

        if (random.random() <= eps):
            best_action = random.randint(0, self.num_actions-1)
        else:
            best_action = self.model.GetAction(s1)

        self.reward = env.Make_Action(best_action, parameter.frame_repeat)

        isterminal=env.IsEpisodeFinished()
        s2 = Preprocess(env.Observation()) if not isterminal else None
        self.memory.Add(s1, best_action, s2, isterminal, self.reward)
        self.LearnFromMemory()
    def Train(self):
        train_scores = []
        env.Reset()
        for iteration in range(1, parameter.how_many_times+1):
            self.perform_learning_step(iteration)
            if(env.IsEpisodeFinished()):
                train_scores.append(self.reward)
                env.Reset()
            if (iteration % parameter.save_each == 0):
                model_name_curr = model_name + "_{:04}".format(int(iteration / parameter.save_each))
                self.saver.save(self.session, model_name_curr)
                #self.saver.save(self.session, DEFAULT_MODEL_SAVEFILE)
                Display_Training(iteration, parameter.how_many_times, train_scores)
                train_scores = []
        env.Reset()
def Test(agent,current_model_name):
    if (test_write_video):
        size = (1, 100)
        fps = 10 #/ frame_repeat
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # cv2.cv.CV_FOURCC(*'XVID')
        out_video = cv2.VideoWriter(Working_Directory + "test1.avi", fourcc, fps, size)

    list_Episode = []
    list_Reward = []
    how_many_times=1
    print('Running Test')
    reward_list=[]
    episode_list=[]
    reward_total = 0
    number_of_episodes = 1
    test=0

    while (test < number_of_episodes):

        if (env.IsEpisodeFinished()):
            env.Reset()
            print("Total reward: {}".format(reward_total))
            reward_list.append(reward_total)
            reward_total = 0
            test=test+1
        state_raw = env.Observation()
        state = Preprocess(state_raw)
        best_action=agent.GetAction(state,current_model_name)
        for _ in range(parameter.frame_repeat):

            if (test_display):
                cv2.imshow("Test", state_raw)
                cv2.waitKey(20)

            if (test_write_video):
                out_video.write(state_raw)

            reward = env.Make_Action(best_action, 1)
            reward_total += reward

            if (env.IsEpisodeFinished()):
                break

            state_raw = env.Observation()
    list_Reward.append(reward_list)
    success=(reward_list.count(1.0)/number_of_episodes)*100
    success_percentate=str(success)+'%'
    parameter.final_test_percentage.append(success)
    print('Success percentage',success_percentate)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="the GPU to use")
    args = parser.parse_args()

    if (args.gpu):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    env = Environment_Audio(scenario_file)
    agent = Agent(env.NumActions())
    reward_list_training=[]

    if(Train_Model):
        for i in range(1,parameter.how_many_times_training+1):
            mean_scores=[]
            print("Training Iteration {}, using {}".format(i,Feature))
            agent.Train()
            print('Mean Scores',mean_scores)
            reward_list_training.append(mean_scores)
        print("Mean List Reward",reward_list_training)

    if(Load_Model):
        how_many_models_to_test=2
        for i in range(1,how_many_models_to_test+1):
            model_name_curr = model_name + "_{:04}".format(int(i))
            print(model_name_curr)
            Test(agent,model_name_curr)
        print(parameter.final_test_percentage)
        final_average_success=np.mean(parameter.final_test_percentage)
        print('Average success percentage using the last {} models is {}%'.format(how_many_models_to_test,final_average_success))