from aurora_lib.network_simulator.pcc.aurora.schedulers import TestScheduler
from aurora_lib.network_simulator.pcc.aurora.aurora import Aurora
from aurora_lib.network_simulator.pcc.aurora.aurora_environment import AuroraEnvironment
from aurora_lib.trace import generate_trace
from aurora_lib.ppo import PPO
import gym
from torch import nn as nn
import torch

NUM_STATES = None

def get_model():
    dummy_trace = generate_trace((10, 10), (2, 2), (2, 2), (50, 50), (0, 0), (1, 1), (0, 0), (0, 0))
    test_scheduler = TestScheduler(dummy_trace)
    env = AuroraEnvironment(trace_scheduler=test_scheduler)
    obs = env.reset()
    global NUM_STATES
    NUM_STATES = env.observation_space.shape[0]
    model = PPO(env, verbose=False)
    model.load_checkpoint("../../applications/aurora/models/model")

    return model.actor

def get_plain_comparative_aurora(if_difference=False):
    class MyModel(nn.ModuleList):
        def __init__(self, device=torch.device("cpu")):
            super(MyModel, self).__init__()
            self.base_model = get_model()
            self.input_size = NUM_STATES
            self.ft = torch.nn.Flatten()
            self.if_difference = if_difference

        def forward(self, obs):
            # input processing
            input1 = obs[:, :, :self.input_size]
            input2 = input1 + obs[:, :, self.input_size:2*self.input_size]

            # the model
            copy1_logits = self.base_model(input1)
            copy2_logits = self.base_model(input2)

            if self.if_difference:
                return self.ft(copy1_logits - copy2_logits)
            else:
                return self.ft(torch.concat((copy1_logits, copy2_logits), dim=1))

    return MyModel()
    
if __name__ == "__main__":
    model = get_plain_comparative_aurora(if_difference=False)
    x = torch.tensor([[[0.1] * NUM_STATES * 2]]) # .to(device="cuda")
    print(model(x))
