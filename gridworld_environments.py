"""
Register gym_minigrid environments using the Gym API.
These environments are slightly simplified from the default gym_minigrid code.
"""

import collections
from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal, OBJECT_TO_IDX, COLOR_TO_IDX
import numpy as np

from gym import spaces
from gym.envs.registration import register
from gym_minigrid.minigrid import *
from gym_minigrid.roomgrid import RoomGrid


CHW = collections.namedtuple('CHW', ('channels', 'height', 'width'))

# Source: These environments come from the gym_minigrid package code.

class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=15,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=math.inf, # 4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

        s = CHW(3, size, size) 
        self.observation_space = spaces.Box(
            low=0,
            high=255,  # TODO
            shape=(s.width, s.height, s.channels),
            dtype='uint8'
        )

        self.states_visited = set()

    def step(self, action):
        obs, reward, done, infos = super().step(action)

        cur_pos = (*self.agent_pos, self.agent_dir)
        self.states_visited.add(cur_pos)

        return self.observation(obs), reward, done, infos

    def observation(self, obs):
        state = obs["image"]

        env = self.unwrapped
        full_grid = self.grid.encode() # todo: Cache this encoding
        full_grid[self.agent_pos[0]][self.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            self.agent_dir
        ])

        return full_grid

    def reset(self):
        obs = super().reset()
        return self.observation(obs)  # ["image"]

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

class FourRoomsEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None, size=None):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(grid_size=size, max_steps=math.inf) # 100)

        s = CHW(3, size, size)  # self.observation_space.spaces["image"]
        self.observation_space = spaces.Box(
            low=0,
            high=255,  # TODO
            shape=(s.width, s.height, s.channels),
            dtype='uint8'
        )

        self.states_visited = set()

    def step(self, action):
        obs, reward, done, infos = super().step(action)

        cur_pos = (*self.agent_pos, self.agent_dir)
        self.states_visited.add(cur_pos)

        return self.observation(obs), reward, done, infos

    def observation(self, obs):
        state = obs["image"]

        env = self.unwrapped
        full_grid = self.grid.encode()  # todo: Cache this encoding
        full_grid[self.agent_pos[0]][self.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            self.agent_dir
        ])

        return full_grid

    def reset(self):
        obs = super().reset()
        return self.observation(obs)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            # assuming random start direction
            self.agent_dir = self._rand_int(0, 4)
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.grid.set(*self._goal_default_pos, goal)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'

class KeyCorridorEnv(RoomGrid):
    """
    A ball is behind a locked door, the key is placed in a
    random room.
    """

    def __init__(
        self,
        num_rows=3,
        obj_type="ball",
        room_size=6,
        seed=None
    ):
        self.obj_type = obj_type

        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            num_cols=num_rows,
            max_steps=math.inf,
            seed=seed,
        )

        size = num_rows * (room_size - 1) + 1
        s = CHW(3, size, size)  # self.observation_space.spaces["image"]
        self.observation_space = spaces.Box(
            low=0,
            high=255,  # TODO
            shape=(s.width, s.height, s.channels),
            dtype='uint8'
        )

        self.states_visited = set()

    def observation(self, obs):
        state = obs["image"]

        env = self.unwrapped
        full_grid = self.grid.encode()  # todo: Cache this encoding
        full_grid[self.agent_pos[0]][self.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            self.agent_dir
        ])

        return full_grid

    def reset(self):
        obs = super().reset()
        return self.observation(obs)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Connect the middle column rooms into a hallway
        for j in range(1, self.num_rows):
            self.remove_wall(1, j, 3)

        # Add a locked door on the bottom right
        # Add an object behind the locked door
        room_idx = self._rand_int(0, self.num_rows)
        door, _ = self.add_door(2, room_idx, 2, locked=True)
        obj, _ = self.add_object(2, room_idx, kind=self.obj_type)

        # Add a key in a random room on the left side
        self.add_object(0, self._rand_int(0, self.num_rows), 'key', door.color)

        # Place the agent in the middle
        self.place_agent(1, self.num_rows // 2)

        # Make sure all rooms are accessible
        self.connect_all()

        self.obj = obj
        self.mission = "pick up the %s %s" % (obj.color, obj.type)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                done = True

        cur_pos = (*self.agent_pos, self.agent_dir)
        self.states_visited.add(cur_pos)

        return self.observation(obs), reward, done, info


class EmptyEnv15x15(EmptyEnv):
    def __init__(self):
        super().__init__(size=15)

class EmptyEnv30x30(EmptyEnv):
    def __init__(self):
        super().__init__(size=30)

register(
    id='MiniGridEmptyEnv15x15Unwrapped-v0',
    entry_point='mlca.gridworld_environments:EmptyEnv15x15',
)

register(
    id='MiniGridEmptyEnv30x30Unwrapped-v0',
    entry_point='mlca.gridworld_environments:EmptyEnv30x30',
)

register(
    id='MiniGridFourRoomsEnvUnwrapped-v0',
    entry_point='mlca.gridworld_environments:FourRoomsEnv',
)

register(
    id='MiniGridKeyCorridorS6R3Unwrapped-v0',
    entry_point='mlca.gridworld_environments:KeyCorridorS6R3',
)


register(
    id='MiniGridEmptyEnv-v0',
    entry_point='mlca.gridworld_environments:EmptyEnv',
)


register(
    id='MiniGridFourRoom-v0',
    entry_point='mlca.gridworld_environments:FourRoomsEnv',
)


register(
    id='MiniGridKeyCorridorEnv-v0',
    entry_point='mlca.gridworld_environments:KeyCorridorEnv',
)
