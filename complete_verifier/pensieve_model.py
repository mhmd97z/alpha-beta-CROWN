from onnx2pytorch import ConvertModel
import onnx 
import torch 
import torch.nn as nn
import pensieve_lib.ppo2 as network


def get_model(size="small", seed=None, number=None):
    assert size in ["small", "mid", "big", "original"]

    if size == "original":
        S_INFO = 6
        S_LEN = 8
        A_DIM = 6
        if seed is not None and number is not None:
            model_path = f"../../applications/pensieve/pensieve_lib/pretrain/seed{seed}/model_{number}.pth"
        else:
            model_path = "../../applications/pensieve/pensieve_lib/pretrain/nn_model_ep_155400.pth"
        actor = network.Network(state_dim=[S_INFO, S_LEN], action_dim=A_DIM)
        actor.load_model(model_path)
        pytorch_model = actor.actor

    else:
        path_to_onnx_model = f"../../applications/pensieve/model/onnx/pensieve_{size}_simple.onnx"
        onnx_model = onnx.load(path_to_onnx_model)
        pytorch_model = ConvertModel(onnx_model)
        
    return pytorch_model


def get_plain_comparative_pensieve(size, seed=None, number=None) -> nn.Sequential:
    base_model = get_model(size, seed=seed, number=number)

    class MyModel(nn.ModuleList):
        def __init__(self, device=torch.device("cpu")):
            super(MyModel, self).__init__()

            self.input_size = 48
            self.ft = torch.nn.Flatten()

            #################
            # Model
            ################# 
            self.base_model = base_model
            
        def forward(self, obs):
            # input processing
            input1 = (obs[:, :, :self.input_size]) # .reshape((-1, 6, 8))
            input2 = (obs[:, :, :self.input_size] + obs[:, :, self.input_size:2*self.input_size]) # .reshape((-1, 6, 8))
            
            # the model
            copy1_logits = self.base_model(input1)
            copy2_logits = self.base_model(input2)
            
            return self.ft(torch.concat((copy1_logits, copy2_logits), dim=1))

    model = MyModel()

    return model
