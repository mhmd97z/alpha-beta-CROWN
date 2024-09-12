from firm_lib.ppo import PPO
from firm_lib.serverless_env import SimEnvironment
from firm_lib.util import *
from torch import nn as nn
import torch

def get_model():
    class ActorNetworkWrapper(nn.Module):
        def __init__(self, input_size=NUM_STATES, hidden_size=HIDDEN_SIZE, 
                output_size=NUM_ACTIONS, base_model=None):
            super(ActorNetworkWrapper, self).__init__()

            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc1.weight.data, self.fc1.bias.data = \
                base_model.fc1.weight.data, base_model.fc1.bias.data

            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc2.weight.data, self.fc2.bias.data = \
                base_model.fc2.weight.data, base_model.fc2.bias.data

            self.fc3 = nn.Linear(hidden_size, output_size)
            self.fc3.weight.data, self.fc3.bias.data = \
                base_model.fc3.weight.data, base_model.fc3.bias.data

            self.relu = nn.ReLU()

        def forward(self, input_):
            # input_ = torch.FloatTensor(input_)
            output = self.relu(self.fc1(input_))
            output = self.relu(self.fc2(output))
            output = self.fc3(output)

            return output

    env = SimEnvironment("../../applications/firm/firm_lib/data/readfile_sleep_imageresize_output.csv")
    function_name = env.get_function_name()
    initial_state = env.reset(function_name)
    folder_path = "../../applications/firm/lib/model/" + str(function_name)
    agent = PPO(env, function_name, folder_path)
    agent.load_checkpoint("../../applications/firm/model/ppo.pth.tar")
    return ActorNetworkWrapper(base_model=agent.actor)

def get_plain_comparative_firm():
    class MyModel(nn.ModuleList):
        def __init__(self, device=torch.device("cpu")):
            super(MyModel, self).__init__()

            self.input_size = NUM_STATES
            self.ft = torch.nn.Flatten()
            
            #################
            # Model
            ################# 
            self.base_model = get_model()
            
            
        def forward(self, obs):
            # input processing
            input1 = obs[:, :, :self.input_size]
            input2 = input1 + obs[:, :, self.input_size:2*self.input_size]

            # the model
            copy1_logits = self.base_model(input1)
            copy2_logits = self.base_model(input2)

            return self.ft(torch.concat((copy1_logits, copy2_logits), dim=1))

    return MyModel()
    
if __name__ == "__main__":
    model = get_plain_comparative_firm()
    x = torch.tensor([[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]]) # .to(device="cuda")
    print(model(x))