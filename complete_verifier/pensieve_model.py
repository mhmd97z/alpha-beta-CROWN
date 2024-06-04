from onnx2pytorch import ConvertModel
import onnx 
import torch 
import torch.nn as nn

def get_model(size="small"):
    assert size in ["small", "mid", "big"]
    
    path_to_onnx_model = f"/home/mzi/sys-rl-verif/VNNComp23_NN4Sys/onnx/pensieve_{size}_simple.onnx"
    onnx_model = onnx.load(path_to_onnx_model)
    pytorch_model = ConvertModel(onnx_model)

    return pytorch_model

def get_params_argmax(input_size):
    
    # Take sum of the input vars
    c01 = torch.zeros([1, 1, input_size+1])
    c01[0][0][0] = 1

    c02 = torch.zeros([1, 1, input_size+1])
    c02[0][0][0] = 1
    c02[0][0][-1] = 1

    return c01, c02

def get_plain_comparative_pensieve() -> nn.Sequential:
    base_model = get_model()

    class MyModel(nn.ModuleList):
        def __init__(self, device=torch.device("cpu")):
            super(MyModel, self).__init__()

            input_size = 48
            self.input_size = input_size
            c01, c02 = get_params_argmax(input_size)
            
            self.ft = torch.nn.Flatten()

            #################
            # Model
            ################# 
            self.base_model = base_model
            
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

    model = MyModel()

    return model
