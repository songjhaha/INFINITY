"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 3D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which takes the 2D spatial + 1D temporal equation directly as a 3D problem

changed using v1000 dataset
"""

#%%
import paddle
import numpy as np
import paddle.nn as nn
from paddle.nn import Conv3D
import paddle.nn.functional as F

import matplotlib.pyplot as plt
from utilities3_paddle import *

import operator
from functools import reduce
from functools import partial

import math
from timeit import default_timer
import scipy.io

paddle.seed(0)
np.random.seed(0)

#%%
import os
def default(value, d):
    '''
    helper taken from https://github.com/lucidrains/linear-attention-transformer
    '''
    return d if value is None else value



# DATA_PATH can be the system environment variable DATAPATH
current_path = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(os.path.dirname(current_path))
MODEL_PATH = default(os.environ.get('MODELPATH'),
                     os.path.join(SRC_ROOT, 'models'))
DATA_PATH = default(os.environ.get('DATAPATH'),
                    os.path.join(SRC_ROOT, 'data'))

#%%
################################################################
# 3d fourier layers
################################################################

class SpectralConv3d(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        temp = paddle.rand([in_channels, out_channels, self.modes1, self.modes2, self.modes3])
        temp1 = self.scale * temp

        self.weights1 = paddle.create_parameter(shape=temp1.shape, dtype=str(temp1.numpy().dtype), default_initializer=paddle.nn.initializer.Assign(temp1))
        self.theta1  = paddle.create_parameter(shape=temp.shape, dtype=str(temp.numpy().dtype), default_initializer=paddle.nn.initializer.Assign(temp))
        self.weights2 = paddle.create_parameter(shape=temp1.shape, dtype=str(temp1.numpy().dtype), default_initializer=paddle.nn.initializer.Assign(temp1))
        self.theta2  = paddle.create_parameter(shape=temp.shape, dtype=str(temp.numpy().dtype), default_initializer=paddle.nn.initializer.Assign(temp))
        self.weights3 = paddle.create_parameter(shape=temp1.shape, dtype=str(temp1.numpy().dtype), default_initializer=paddle.nn.initializer.Assign(temp1))
        self.theta3  = paddle.create_parameter(shape=temp.shape, dtype=str(temp.numpy().dtype), default_initializer=paddle.nn.initializer.Assign(temp))
        self.weights4 = paddle.create_parameter(shape=temp1.shape, dtype=str(temp1.numpy().dtype), default_initializer=paddle.nn.initializer.Assign(temp1))
        self.theta4  = paddle.create_parameter(shape=temp.shape, dtype=str(temp.numpy().dtype), default_initializer=paddle.nn.initializer.Assign(temp))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return paddle.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = paddle.fft.rfftn(x, axes=(-3,-2,-1))

        # Multiply relevant Fourier modes
        out_ft = paddle.zeros(shape=[batchsize, self.out_channels, x.shape[-3], x.shape[-2], x.shape[-1]//2 + 1], dtype="complex64")

        # WorkAround unsupported set_value complex feature
        # Left top 8*8 submatrix
        R1 = self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1*paddle.cos(self.theta1*2*math.pi)+self.weights1*paddle.sin(self.theta1*2*math.pi)*1j)
        # Left bottom 8*8
        R2 = self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2*paddle.cos(self.theta2*2*math.pi)+self.weights2*paddle.sin(self.theta2*2*math.pi)*1j)
        # Right top 8*8
        R3 = self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3*paddle.cos(self.theta3*2*math.pi)+self.weights3*paddle.sin(self.theta3*2*math.pi)*1j)
        # Right bottom 8*8
        R4 = self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4*paddle.cos(self.theta4*2*math.pi)+self.weights4*paddle.sin(self.theta4*2*math.pi)*1j)

        # First 8*64 grids,[4, 20, 8, 64, 3]
        first_8_64 = paddle.concat(x=[R1, out_ft[:,:,:self.modes1, self.modes2:-self.modes2,:self.modes3], R3], axis=3)
        # Thrid 8*64 [4, 20, 8, 64, 3]
        third_8_64 = paddle.concat(x=[R2, out_ft[:,:,-self.modes1:, self.modes2:-self.modes2,:self.modes3], R4], axis=3)
        # Total 64*64 [4, 20, 64, 64, 3]
        total_64_64 = paddle.concat(x=[first_8_64, out_ft[:,:, self.modes1:-self.modes1,:, :self.modes3], third_8_64], axis=2)
        # All 64*64*3
        final_R=paddle.concat(x=[total_64_64, out_ft[:,:,:,:,-self.modes3:]], axis=4)

        #Return to physical space
        x = paddle.fft.irfftn(final_R, s=(x.shape[-3], x.shape[-2], x.shape[-1]))
        return x

class FNO3d(paddle.nn.Layer):
    def __init__(self, modes1, modes2, modes3, width):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(13, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = Conv3D(self.width, self.width, 1)
        self.w1 = Conv3D(self.width, self.width, 1)
        self.w2 = Conv3D(self.width, self.width, 1)
        self.w3 = Conv3D(self.width, self.width, 1)

        # self.bn0 = torch.nn.BatchNorm3d(self.width)
        # self.bn1 = torch.nn.BatchNorm3d(self.width)
        # self.bn2 = torch.nn.BatchNorm3d(self.width)
        # self.bn3 = torch.nn.BatchNorm3d(self.width)

        self.fc1 = paddle.nn.Linear(self.width, 128)
        self.fc2 = paddle.nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = paddle.transpose(x, perm=[0, 4, 1, 2, 3])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = nn.GELU()(x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = nn.GELU()(x)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = nn.GELU()(x)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = paddle.transpose(x, perm=[0, 2, 3, 4, 1])
        x = self.fc1(x)
        x = nn.GELU()(x)
        x = self.fc2(x)
        return x


################################################################
# configs
################################################################

#TRAIN_PATH = TEST_PATH = os.path.join(DATA_PATH, 'ns_V1e-3_N5000_T50.mat')
#TRAIN_PATH = TEST_PATH = os.path.join('/home/xianghui01/FNO-11-22/fourier_neural_operator-master/data/ns_V1e-3_N5000_T50.mat')
#TRAIN_PATH = TEST_PATH = os.path.join('/home/xianghui01/data/ns_V1e-3_N5000_T50.mat')
TRAIN_PATH = TEST_PATH = os.path.join('/home/xianghui01/NavierStokes_V1e-5_N1200_T20.mat')

ntrain = 1000
ntest = 200

batch_size = 10

epochs = 500
learning_rate = 0.0025
scheduler_step = 100
scheduler_gamma = 0.5

path = 'paddle_ns_V1e-5_N1200_20'
# path = 'ns_fourier_V100_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path

t1 = default_timer()

sub = 1
S = 64 // sub
T_in = 10
T = 10


modes = 8
#modes_t = 8
modes_t = T//4+1 # modes for time, original FNO3D chose modes_t = modes which is wrong
width = 20

#%%
################################################################
# load data
################################################################

reader = MatReader(TRAIN_PATH)
# the data size is (5000, 64, 64, 50)
# 5000 is the number of training samples
# 64x64 is the size of the spatial domain
# 50 is the number of time steps
#%%
train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]
train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]

reader = MatReader(TEST_PATH)
test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in]
test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]

print(train_u.shape)
print(test_u.shape)
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

#%%
a_normalizer = UnitGaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

y_normalizer = UnitGaussianNormalizer(train_u)
train_u = y_normalizer.encode(train_u)

train_a = train_a.reshape([ntrain,S,S,1,T_in]).expand([ntrain,S,S,T,T_in])
test_a = test_a.reshape([ntest,S,S,1,T_in]).expand([ntest,S,S,T,T_in])

# pad locations (x,y,t)
gridx = paddle.to_tensor(np.linspace(0, 1, S), dtype="float32")
gridx = gridx.reshape([1, S, 1, 1, 1]).expand([1, S, S, T, 1])
gridy = paddle.to_tensor(np.linspace(0, 1, S), dtype="float32")
gridy = gridy.reshape([1, 1, S, 1, 1]).expand([1, S, S, T, 1])
gridt = paddle.to_tensor(np.linspace(0, 1, T+1)[1:], dtype="float32")
gridt = gridt.reshape([1, 1, 1, T, 1]).expand([1, S, S, T, 1])

train_a = paddle.concat([gridx.expand([ntrain,S,S,T,1]), gridy.expand([ntrain,S,S,T,1]),
                       gridt.expand([ntrain,S,S,T,1]), train_a], axis=-1)
test_a = paddle.concat([gridx.expand([ntest,S,S,T,1]), gridy.expand([ntest,S,S,T,1]),
                       gridt.expand([ntest,S,S,T,1]), test_a], axis=-1)

print(train_a.shape)
print(test_a.shape)

train_loader = paddle.io.DataLoader(paddle.io.TensorDataset([train_a, train_u]), batch_size=batch_size, shuffle=True)
test_loader = paddle.io.DataLoader(paddle.io.TensorDataset([test_a, test_u]), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)

#%%
sample = next(iter(train_loader))
for s in sample:
    print(s.shape)

#%%
################################################################
# training and evaluation
################################################################
model = FNO3d(modes, modes, modes_t, width)
# model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

scheduler = paddle.optimizer.lr.StepDecay(learning_rate=learning_rate, step_size=scheduler_step, gamma=scheduler_gamma)
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=learning_rate, weight_decay=1e-5)

T_max = len(train_loader) * epochs / 2
#scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=learning_rate, T_max=T_max, verbose=False)
#scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=10, verbose=False)
#optimizer = paddle.optimizer.SGD(learning_rate=scheduler, parameters=model.parameters())
'''
scheduler = paddle.optimizer.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                        div_factor=1e4, final_div_factor=1e4,
                        steps_per_epoch=len(train_loader), epochs=epochs)
'''

myloss = LpLoss(size_average=False)
y_normalizer.cuda()

#%%
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.clear_grad()
        out = paddle.reshape(model(x), [batch_size, S, S, T])

        mse = F.mse_loss(out, y, reduction='mean')
        # mse.backward()

        y = y_normalizer.decode(y)
        out = y_normalizer.decode(out)
        l2 = myloss(paddle.reshape(out, [batch_size, -1]), paddle.reshape(y, [batch_size, -1]))
        l2.backward()

        optimizer.step()
        #scheduler.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with paddle.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = paddle.reshape(model(x), [batch_size, S, S, T])
            out = y_normalizer.decode(out)
            test_l2 += myloss(paddle.reshape(out, [batch_size, -1]), paddle.reshape(y, [batch_size, -1])).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2, test_l2)
    paddle.save(model.state_dict(), path_model)

pred = paddle.zeros(test_u.shape)
index = 0
test_loader = paddle.io.DataLoader(paddle.io.TensorDataset([test_a, test_u]), batch_size=1, shuffle=False)
with paddle.no_grad():
    for x, y in test_loader:
        test_t1 = default_timer()
        test_l2 = 0
        x, y = x.cuda(), y.cuda()

        out = model(x).reshape([1,S,S,T])
        out = y_normalizer.decode(out)
        pred[index] = out.reshape([S, S, T])

        test_l2 += myloss(out.reshape([1, -1]), y.reshape([1, -1])).item()
        test_t2 = default_timer()
        print(index, test_t2-test_t1, test_l2)
        index = index + 1

scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})
