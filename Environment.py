from __future__ import division
from __future__ import print_function

import itertools as it

from vizdoom import *
from pydub import AudioSegment
from pydub.playback import play
from ExtractFeatures import Extract_Features

import scipy
target_position_x = 100
target_position_y = 100

Extract=Extract_Features

class Environment(object):
    prev_reward=None
    def __init__(self, scenario_path):

        print("Initializing the doom.")
        self.game = DoomGame()
        self.game.set_doom_scenario_path(scenario_path)
        self.game.set_doom_map("map01")
        self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.set_render_hud(True) # False
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(False)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        
        self.game.add_available_button(Button.TURN_LEFT)
        self.game.add_available_button(Button.TURN_RIGHT)
        self.game.add_available_button(Button.MOVE_FORWARD)
        self.game.add_available_button(Button.MOVE_BACKWARD)
        
        self.game.set_episode_timeout(2100)
        self.game.set_episode_start_time(14)
        self.game.set_window_visible(True)
        self.game.set_sound_enabled(False)
        self.game.set_living_reward(0)
        self.game.set_mode(Mode.PLAYER)

        self.game.set_labels_buffer_enabled(False)
        self.game.clear_available_game_variables()
        self.game.add_available_game_variable(GameVariable.POSITION_X)
        self.game.add_available_game_variable(GameVariable.POSITION_Y)
        self.game.add_available_game_variable(GameVariable.POSITION_Z)

        self.game.init()
        print("Doom is initialized.")

        n = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=n)]
        self.num_actions = len(self.actions)

    def NumActions(self):
        return self.num_actions

    def Reset(self):
        self.game.new_episode()

    def Make_Action(self, action, frame_repeat):
        action=self.MapActions(action)
        return self.game.make_action(self.actions[action], frame_repeat)

    def IsEpisodeFinished(self):
        return (self.game.is_episode_finished())

    def Observation(self):
        s = self.game.get_state()
        player_position_x = s.game_variables[0]
        player_position_y = s.game_variables[1]
        return (Extract.Extract_Samples(self, player_position_x, player_position_y))

    def MapActions(self, action_raw):
        return action_raw
