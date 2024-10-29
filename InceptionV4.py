import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
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
        
        #print(torch.cat((x5, x6), 1).shape)
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

        #print("Reduction A")
        #print(torch.cat((x1, x2, x3), 1).shape)
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

        #print(torch.cat((x1, x2, x3, x4), 1).shape)
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

        #print("Reduction B")
        #print(torch.cat((x1, x2, x3), 1).shape)
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
        
        #print(x.shape)
        #print(x1.shape)
        #print(x2.shape)
        #print(x3.shape)
        #print(x4.shape)
        #print(x5.shape)
        #print(x6.shape)
        #print(x7.shape)
        #print(x8.shape)

        print(torch.cat((x1, x3, x4, x6, x7, x8), 1).shape)

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

        #print(self.inception_c.shape)
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

# Create a random tensor simulating a batch of images (batch_size, channels, height, width)
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