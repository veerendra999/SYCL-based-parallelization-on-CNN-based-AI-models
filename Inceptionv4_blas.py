import torch
import torch.nn as nn
import torch.nn.functional as F
import ctypes
import numpy as np
import time

global totalTimeTaken
totalTimeTaken = 0
class Kn2row2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, out_channels,  bias=None, stride=1, padding=0):

        input_flatten = input.view(input.size(0), -1)
        weight_flatten = weight.view(weight.size(0), -1)

        # Calling the convolution function
        print(input.size(2), weight.size(2), padding, stride)
        if type(padding) == tuple:
            padding_x = padding[0]
            padding_y = padding[1]
        else:
            padding_x = padding
            padding_y = padding
            
        # different dimensions for rectangular kernel sizes
        op_dim_x = (input.size(2) - weight.size(2) + (2 * padding_x)) // stride 
        op_dim_x += 1
        op_dim_y = (input.size(2) - weight.size(3) + (2 * padding_y)) // stride 
        op_dim_y += 1
        output_array = np.empty((op_dim_x, op_dim_y, out_channels), dtype=np.float32)
        
        cpp_lib = ctypes.CDLL('./kn2row.so')  # please find the so file after the compile command in dir
        
        cpp_lib.kn2row.restype = ctypes.c_double
        timetaken_microseconds = cpp_lib.kn2row(input_flatten.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       input.size(2), input.size(3), input.size(1), out_channels,
                       weight_flatten.numpy().ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                       weight.size(2), weight.size(3), stride, padding_x, padding_y, op_dim_x, op_dim_y,
                       output_array.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        
        
        global totalTimeTaken
        totalTimeTaken += timetaken_microseconds

        result_tensor = torch.from_numpy(output_array.reshape((1, out_channels, op_dim_x, op_dim_y)))
        return result_tensor

class Kn2row2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
        super(Kn2row2d, self).__init__()
        
        # check for kernel sizes for inceptionv4 architecture
        if type(kernel_size) is tuple:
            k_row = kernel_size[0]
            k_col = kernel_size[1]
        else:
            k_row = kernel_size
            k_col = kernel_size
            
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, k_row, k_col))
        self.bias = nn.Parameter(torch.randn(out_channels))
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels

    def forward(self, x):  # Include groups parameter in forward
        return Kn2row2dFunction.apply(x, self.weight, self.out_channels,  self.bias, self.stride, self.padding)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        
        #custom Kn2row2d convolution function instead of basicConv2d
        self.conv = Kn2row2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionStem(nn.Module):
    def __init__(self, in_channels):
        super(InceptionStem, self).__init__()
        self.conv1 = BasicConv2d(in_channels, 32, kernel_size=3, stride=2)
        self.conv2 = BasicConv2d(32, 32, kernel_size=3)
        self.conv3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.branch1 = nn.MaxPool2d(3, stride=2)
        self.branch2 = nn.Sequential(  
            BasicConv2d(64, 96, kernel_size=3, stride=2),
            #BasicConv2d(96, 96, kernel_size=3),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, ),
            BasicConv2d(64, 96, kernel_size=3),
        )
        self.branch4 = nn.Sequential(
            #nn.AvgPool2d(3, stride=2),
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3),
        )
        self.branch5 = nn.MaxPool2d(2, stride=2)
        self.branch6 = nn.Sequential(
            BasicConv2d(192, 192, kernel_size=3, stride=2),  
        )
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(torch.cat((x1, x2), 1))
        x4 = self.branch4(torch.cat((x1, x2), 1))
        x5 = self.branch5(torch.cat((x3, x4), 1))
        x6 = self.branch6(torch.cat((x3, x4), 1))
        
        return torch.cat((x5, x6), 1)
        

class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1 = BasicConv2d(in_channels, 96, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, 96, kernel_size=1),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        
        return torch.cat((x1, x2, x3, x4), 1)

class ReductionA(nn.Module):
    def __init__(self, in_channels):
        super(ReductionA, self).__init__()
        self.branch1 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=3, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2),
        )
        self.branch3 = nn.MaxPool2d(3, stride=2)
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        return torch.cat((x1, x2, x3), 1)


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(192, 224, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(224, 224, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(224, 256, kernel_size=(1, 7), padding=(0, 3)),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, 128, kernel_size=1),
        )
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        
        return torch.cat((x1, x2, x3, x4), 1)

class ReductionB(nn.Module):
    def __init__(self, in_channels):
        super(ReductionB, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(256, 320, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2),
        )
        self.branch3 = nn.MaxPool2d(3, stride=2)
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        return torch.cat((x1, x2, x3), 1)

class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super(InceptionC, self).__init__()
        self.branch1 = BasicConv2d(in_channels, 256, kernel_size=1)
        self.branch2 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3 = BasicConv2d(384, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch4 = BasicConv2d(384, 256, kernel_size=(3, 1), padding=(1, 0))   
  
        self.branch5 = nn.Sequential(
            BasicConv2d(in_channels, 384, kernel_size=1),
            BasicConv2d(384, 448, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(448, 512, kernel_size=(1, 3), padding=(0, 1)),
        )
        self.branch6 = BasicConv2d(512, 256, kernel_size=(1, 3), padding=(0, 1))
        self.branch7 = BasicConv2d(512, 256, kernel_size=(3, 1), padding=(1, 0))
        
        self.branch8 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_channels, 256, kernel_size=1),
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x2)
        x4 = self.branch4(x2)
        x5 = self.branch5(x)
        x6 = self.branch6(x5)
        x7 = self.branch7(x5)
        x8 = self.branch8(x)

        return torch.cat((x1, x3, x4, x6, x7, x8), 1) 


 # InceptionV4 Model
class InceptionV4(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionV4, self).__init__()
        self.stem = InceptionStem(3)
        # Add simplified Inception and Reduction blocks
        self.inception_a = InceptionA(384)
        self.reduction_a = ReductionA(384)
        self.inception_b = InceptionB(1024)
        self.reduction_b = ReductionB(1024)
        self.inception_c = InceptionC(1536)
        # Adjusted linear layer to match the output
        self.fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Instantiate the model
inceptionv4 = InceptionV4()

# random tensor simulating a batch of images (batch_size, channels, height, width)
# Let's assume a batch size of 4
random_input = torch.randn(4, 3, 299, 299)

# Move the model to evaluation mode
inceptionv4.eval()

# Measure inference time
start_time = time.time()

# Disable gradient computation for inference
with torch.no_grad():
    # Perform inference
    outputs = inceptionv4(random_input)

end_time = time.time()
inference_time = end_time - start_time

print("Inference Time: {:.6f} seconds".format(inference_time))
