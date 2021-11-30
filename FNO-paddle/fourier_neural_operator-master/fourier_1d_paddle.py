# _*_ conding:utf-8 _*_
# Team: CASIC 
# Member: hly
# Date & Time: 2021/11/3 22:28
# Filename: fourier_1d_paddle.py
# Tool: PyCharm

import math
import numpy as np
import paddle
import paddle.nn as nn 
import paddle.fft
#import x2paddle
from paddle.nn import Conv1D
import paddle.nn.functional as F          # import torch.nn.functional as F #  paddle.nn.functional
from timeit import default_timer          # ?????
from utilities3_paddle import *                  # ?????
#from  x2paddle import torch2paddle
import scipy
from scipy import io
# from Adam import Adam 

paddle.seed(0)
np.random.seed(0)


################################################################
#  1d fourier layer
################################################################
class SpectralConv1D(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1D, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        print("testing:",in_channels, out_channels, self.modes1)
        temp1 = self.scale * paddle.rand([in_channels, out_channels, self.modes1])
        temp2 = paddle.rand([in_channels, out_channels, self.modes1])
        self.weights1 = paddle.create_parameter(shape=temp1.shape, dtype=str(temp1.numpy().dtype), default_initializer=paddle.nn.initializer.Assign(temp1))
        self.theta  = paddle.create_parameter(shape=temp2.shape, dtype=str(temp2.numpy().dtype), default_initializer=paddle.nn.initializer.Assign(temp2))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return paddle.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfft(x)

        # Multiply relevant Fourier modes
        # out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft = paddle.zeros(shape=[batchsize, self.out_channels, x.shape[-1]//2 +1], dtype="complex64")  
        #out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1*paddle.cos(self.theta*2*math.pi)+self.weights1*paddle.sin(self.theta*2*math.pi)*1j)

        #Return to physical space
        #x = paddle.fft.irfft(out_ft, n=x.shape(-1))
        x = paddle.fft.irfft(self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1*paddle.cos(self.theta*2*math.pi)+self.weights1*paddle.sin(self.theta*2*math.pi)*1j), n=x.shape[-1])
        return x


class FNO1d(paddle.nn.Layer):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2, self.width)  # input channel is 2: (a(x), x)
        
        print("SpecCon1d: flag-start")
        self.conv0 = SpectralConv1D(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1D(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1D(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1D(self.width, self.width, self.modes1)
        
        self.w0 = Conv1D(self.width, self.width, 1)
        self.w1 = Conv1D(self.width, self.width, 1)
        self.w2 = Conv1D(self.width, self.width, 1)
        self.w3 = Conv1D(self.width, self.width, 1)

        self.fc1 = paddle.nn.Linear(self.width, 128)
        self.fc2 = paddle.nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape)
        x = paddle.concat([x, grid], axis=-1)
        x = self.fc0(x)
        x = paddle.transpose(x, perm=[0, 2, 1])
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = paddle.transpose(x, perm=[0, 2, 1])
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape):
        batchsize, size_x = shape[0], shape[1]
        gridx = paddle.to_tensor(np.linspace(0, 1, size_x), dtype="float32")
        import pdb
        pdb.set_trace()
        temp = gridx.reshape([1, size_x, 1])
        gridx = temp.expand( [batchsize*temp.shape[0]]+ temp.shape[1:] )
        # gridx = paddle.cast()
        return gridx

################################################################
#  configurations
################################################################
ntrain = 1000
ntest = 100

sub = 2**3 #subsampling rate
h = 2**13 // sub #total grid size divided by the subsampling rate
s = h

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 50
gamma = 0.5

modes = 16
width = 64


################################################################
# read data
################################################################
# Data is of the shape (number of samples, grid size)
dataloader = scipy.io.loadmat('/home/xianghui01/data/burgers_data_R10.mat')
# x_data = dataloader.read_field('a')[:,::sub]
# y_data = dataloader.read_field('u')[:,::sub]
x_data = dataloader['a'][:,::sub]
x_data = x_data.astype(np.float32)
x_data = paddle.to_tensor(x_data)
y_data = dataloader['u'][:,::sub]
y_data = y_data.astype(np.float32)
y_data = paddle.to_tensor(y_data)

x_train = x_data[:ntrain,:]
y_train = y_data[:ntrain,:]
x_test = x_data[-ntest:,:]
y_test = y_data[-ntest:,:]
x_train = x_train.reshape([ntrain,s,1])
x_test = x_test.reshape([ntest,s,1])

#x_data = dataloader.read_field('a')[:,::sub]
#y_data = dataloader.read_field('u')[:,::sub]
#print("x_data_type",x_data.dtype)
#x_train = x_data[:ntrain,:]
#y_train = y_data[:ntrain,:]
#x_test = x_data[-ntest:,:]
#y_test = y_data[-ntest:,:]
#print("x_data_shape,x_train_shape:",x_data.shape,x_train.shape)
#x_train = x_train.reshape([ntrain,s,1])
#x_test = x_test.reshape([ntest,s,1])

train_loader = paddle.io.DataLoader(paddle.io.TensorDataset([x_train, y_train]), batch_size=batch_size, shuffle=True)
test_loader = paddle.io.DataLoader(paddle.io.TensorDataset([x_test, y_test]), batch_size=batch_size, shuffle=False)

# model
model = FNO1d(modes, width)

################################################################
# training and evaluation
################################################################
# optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

scheduler = paddle.optimizer.lr.StepDecay(learning_rate, step_size=step_size, gamma=gamma)
optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters(),weight_decay=1e-4)

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.clear_grad()
        out = model(x)

        mse = F.mse_loss(paddle.reshape(out, [batch_size, -1]), paddle.reshape(y, [batch_size, -1]), reduction='mean')
        l2 = myloss(paddle.reshape(out, [batch_size, -1]), paddle.reshape(y, [batch_size, -1]))
        l2.backward()  # use the l2 relative loss

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with paddle.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            test_l2 += myloss(paddle.reshape(out, [batch_size, -1]), paddle.reshape(y, [batch_size, -1])).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2, test_l2)

paddle.save(model, 'model/ns_fourier_burgers')
pred = paddle.zeros(shape=y_test.shape)
index = 0
test_loader = paddle.io.DataLoader(paddle.io.TensorDataset([x_test,y_test]), batch_size=1, shuffle=False)
with paddle.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.cuda(), y.cuda()

        out = paddle.reshape(model(x), [-1])
        pred[index] = out

        test_l2 += myloss(paddle.reshape(out,[1, -1]), paddle.reshape(y, [1, -1])).item()
        print(index, test_l2)
        index = index + 1

 scipy.io.savemat('pred/burger_test.mat', mdict={'pred': pred.cpu().nu64y()})
