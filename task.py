# Hello World program in Python
    
import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=2., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.target_done=False

        # Goal, but will be overriding with a target_pos as desired height.
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.target_height=self.target_pos[2]
        #print("target height={}" .format(self.target_height)) 
   

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        #old_reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #defining the reward for take off, reaching target pos
        #pose < target_ht->R=-difference,pose.ht.=target_ht->R=+10.0
        #reward = -min(abs(self.target_height - self.sim.pose[2]), self.target_height + 10)
        reward=np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        #print("old_reward={},new reward={}" .format(old_reward,reward)) 
        #if self.sim.pose[2] >= self.target_height:
        #   self.target_done=True
        #   reward += 10.0
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            if not done:
                done=self.target_done 
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state