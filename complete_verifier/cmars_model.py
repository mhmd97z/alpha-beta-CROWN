import torch
import torch.nn as nn
from cmars_lib.config_mappol import get_config
import cmars_lib.mdp_config as mdp_config
from cmars_lib.util import *
from cmars_lib.cnn import *
from cmars_lib.act import *
from cmars_lib.mlp import *
from cmars_lib.distributions import *
from gym import spaces

parser = get_config()
parser.add_argument("--add_move_state", action='store_true', default=False)
parser.add_argument("--add_local_obs", action='store_true', default=False)
parser.add_argument("--add_distance_state", action='store_true', default=False)
parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
parser.add_argument("--add_agent_id", action='store_true', default=False)
parser.add_argument("--add_visible_state", action='store_true', default=False)
parser.add_argument("--add_xy_state", action='store_true', default=False)
parser.add_argument("--use_state_agent", action='store_true', default=False)
parser.add_argument("--use_mustalive", action='store_false', default=True)
parser.add_argument("--add_center_xy", action='store_true', default=False)
parser.add_argument("--use_single_network", action='store_true', default=False)
all_args = parser.parse_known_args()[0]

class R_Actor(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float64, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)

        actor_features = self.base(obs)
        action_ligits_probs = self.act(actor_features, available_actions, deterministic)

        return action_ligits_probs

def get_model(layer_count, hidden_size, action_count):
    # set configs
    assert layer_count in [1, 2, 3, 4, 5, 6, 7, 8]
    assert hidden_size in [32, 64, 128]
    assert action_count in [15, 30, 80, 140]

    all_args.layer_N = layer_count
    all_args.hidden_size = hidden_size
    n_prbs = action_count
    models_path = f"../../applications/cmars/models/output_{action_count}/h{hidden_size}/N{layer_count}/actor_type_embb.pt"

    act_space = spaces.Discrete(n_prbs)
    obs_space = spaces.Box(low=0, high=10e6, shape=(mdp_config.EMBB_LOCAL_OBS_VAR_COUNT+mdp_config.AUG_LOCAL_STATE_VAR_COUNT,))

    # base policy
    device = torch.device('cpu')
    model = R_Actor(all_args, obs_space, act_space)
    model.load_state_dict(torch.load(models_path, map_location=device))

    # remove softmax
    class CMARS_Actor_Wrapper(nn.ModuleList):
        def __init__(self, model, device=torch.device("cpu")):
            super(CMARS_Actor_Wrapper, self).__init__()        
            self.to(device)

            self.af = nn.ReLU()
            self.lin1 = nn.Linear(mdp_config.AUG_LOCAL_STATE_VAR_COUNT + mdp_config.EMBB_LOCAL_OBS_VAR_COUNT, all_args.hidden_size)
            
            self.midlayers = []
            for i in range(all_args.layer_N):
                self.midlayers.append(nn.Linear(all_args.hidden_size, all_args.hidden_size))

            for iter, item in enumerate(self.midlayers):
                item.weight.data = model.base.mlp.fc2[iter][0].weight.data
                item.bias.data = model.base.mlp.fc2[iter][0].bias.data

            for i in range(all_args.layer_N):
                setattr(self, "lin{}".format(i+2), self.midlayers[i])

            self.out = nn.Linear(all_args.hidden_size, n_prbs)
            self.out.weight.data = model.act.action_out.linear.weight.data
            self.out.bias.data = model.act.action_out.linear.bias.data

        def forward(self, obs):
            obs = self.af(self.lin1(obs))

            for item in self.midlayers:
                obs = self.af(item(obs))    

            logits = self.out(obs)

            return logits

    embb_cmars_wrapper = CMARS_Actor_Wrapper(model)

    return embb_cmars_wrapper

def get_params_argmax(input_size):
    
    # Take sum of the input vars
    c01 = torch.zeros([1, 1, input_size+1])
    c01[0][0][0] = 1

    c02 = torch.zeros([1, 1, input_size+1])
    c02[0][0][0] = 1
    c02[0][0][-1] = 1

    return c01, c02

def get_plain_comparative_cmars(layer_count=1, hidden_size=32, action_count=15):
    class MyModel(nn.ModuleList):
        def __init__(self, device=torch.device("cpu")):
            super(MyModel, self).__init__()

            input_size = 19
            self.input_size = input_size
            c01, c02 = get_params_argmax(input_size)
            
            self.ft = torch.nn.Flatten()

            #################
            # Model
            ################# 
            self.base_model = get_model(layer_count, hidden_size, action_count)
            
            #################
            # Input summation
            #################
            self.input_conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=input_size+1)
            self.input_conv1.weight = torch.nn.Parameter(c01, requires_grad=True)
            self.input_conv1.bias = torch.nn.Parameter(torch.zeros_like(self.input_conv1.bias, requires_grad=True))
            
            self.input_conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=input_size+1)
            self.input_conv2.weight = torch.nn.Parameter(c02, requires_grad=True)
            self.input_conv2.bias = torch.nn.Parameter(torch.zeros_like(self.input_conv2.bias, requires_grad=True))            
            
        def forward(self, obs):
            # input processing
            input1 = self.input_conv1(obs)
            input2 = self.input_conv2(obs)
            
            # the model
            copy1_logits = self.base_model(input1)
            copy2_logits = self.base_model(input2)
            
            return self.ft(torch.concat((copy1_logits, copy2_logits), dim=1))

    return MyModel()

if __name__ == "__main__":
    print(get_model())