from math import pi
from PIL import Image
from torchvision.transforms import ToTensor
import cv2
import numpy as np
import torch
import torch.nn.functional as functional
import matlab.engine as engine
import matlab


device = torch.device('cpu')
eng = engine.start_matlab()


def phasecong2Matlab(inputTensor):
    inputTensor = inputTensor.cpu()
    inputTensor = inputTensor.squeeze(0).squeeze(0)
    inputTensor = inputTensor.to(torch.int) 
    return eng.phasecong2(matlab.double(inputTensor.tolist()))


def to4Dim(inputTensor):
    needs = 4 - inputTensor.ndim
    if needs >= 0:
        for _ in range(needs):
            inputTensor = inputTensor.unsqueeze(0)
    else:
        for _ in range(-needs):
            inputTensor = inputTensor.squeeze(0)
    return inputTensor.to(device)


def FeatureSIM(imageRef, imageDis):
    assert type(imageRef) == torch.Tensor, 'input image should be torch Tensor'
    assert type(imageDis) == torch.Tensor, 'input image should be torch Tensor'
    assert imageRef.shape == imageDis.shape, 'two input images should be of the same size'
    if imageRef.ndim == 2:
        print('dealing with gray images')
        print('not supporting gray images currently')
        raise ValueError
    if imageRef.ndim == 4:
        imageRef = imageRef.squeeze(0)
        imageDis = imageDis.squeeze(0)
    channels, rows, cols = imageRef.shape
    if imageRef.max() <= 1:
        imageRef = imageRef * 255
        imageDis = imageDis * 255

    if channels == 3:
        Y1 = 0.299 * imageRef[0] + 0.587 * imageRef[1] + 0.114 * imageRef[2]
        Y2 = 0.299 * imageDis[0] + 0.587 * imageDis[1] + 0.114 * imageDis[2]
        I1 = 0.596 * imageRef[0] - 0.274 * imageRef[1] - 0.322 * imageRef[2]
        I2 = 0.596 * imageDis[0] - 0.274 * imageDis[1] - 0.322 * imageDis[2]
        Q1 = 0.211 * imageRef[0] - 0.523 * imageRef[1] + 0.312 * imageRef[2]
        Q2 = 0.211 * imageDis[0] - 0.523 * imageDis[1] + 0.312 * imageDis[2]
        I2 = torch.tensor(I2, dtype=torch.float64)
        Q2 = torch.tensor(Q2, dtype=torch.float64)
        Y2 = torch.tensor(Y2, dtype=torch.float64)
        Y1 = torch.tensor(Y1, dtype=torch.float64)
    else:
        Y1 = imageRef.squeeze(0)
        Y2 = imageDis.squeeze(0)

    # Downsample the image
    minDimension = rows if cols > rows else cols
    F = max(1, round(minDimension / 256))
    aveKernel = torch.empty(1, 1, F, F).fill_(1/(F**2))
    aveKernel = torch.tensor(aveKernel, dtype=torch.float64).to(device)
    aveI1 = functional.conv2d(to4Dim(I1), aveKernel, padding=F//2)
    aveI2 = functional.conv2d(to4Dim(I2), aveKernel, padding=F//2)
    I1 = aveI1[:, :, 1:rows:F, 1:cols:F]
    I2 = aveI2[:, :, 1:rows:F, 1:cols:F]
    
    aveQ1 = functional.conv2d(to4Dim(Q1), aveKernel, padding=F//2)
    aveQ2 = functional.conv2d(to4Dim(Q2), aveKernel, padding=F//2)
    Q1 = aveQ1[:, :, 1:rows:F, 1:cols:F]
    Q2 = aveQ2[:, :, 1:rows:F, 1:cols:F]

    aveY1 = functional.conv2d(to4Dim(Y1), aveKernel, padding=F//2)
    aveY2 = functional.conv2d(to4Dim(Y2), aveKernel, padding=F//2)
    Y1 = aveY1[:, :, 1:rows:F, 1:cols:F]
    Y2 = aveY2[:, :, 1:rows:F, 1:cols:F]
    
    # calculate the phase congruency maps

    PC1 = phasecong2Matlab(Y1)
    PC2 = phasecong2Matlab(Y2)
    
    PC1 = torch.Tensor(PC1).to(torch.float).unsqueeze(0).unsqueeze(0)
    PC2 = torch.Tensor(PC2).to(torch.float).unsqueeze(0).unsqueeze(0)
    PC1 = PC1.to(device)
    PC2 = PC2.to(device)

    # calculate the gradient map

    dx = torch.Tensor([[[[3, 0, -3],
                       [10, 0, -10],
                       [3, 0, -3]]]])/16
    dx = torch.tensor(dx, dtype=torch.float64)
    dy = torch.Tensor([[[[3, 10, 3],
                       [0, 0, 0],
                       [-3, -10, -3]]]])/16
    dy = torch.tensor(dy, dtype=torch.float64)
    IxY1 = functional.conv2d(Y1, to4Dim(dx), padding = 1)
    IyY1 = functional.conv2d(Y1, to4Dim(dy), padding = 1)

    gradientMap1 = torch.sqrt(IxY1**2 + IyY1**2)
    
    IxY2 = functional.conv2d(Y2, to4Dim(dx), padding = 1)
    IyY2 = functional.conv2d(Y2, to4Dim(dy), padding = 1)

    gradientMap2 = torch.sqrt(IxY2**2 + IyY2**2)

    # calculate FSIM
    T1 = 0.85
    T2 = 160
    
    PCSimMatrix = (2 * PC1 * PC2 + T1) / (PC1**2 + PC2**2 + T1)
    gradientSimMatrix = (2* gradientMap1 * gradientMap2 + T2) /(gradientMap1**2 + gradientMap2**2 + T2)
    PCm = torch.max(PC1, PC2)
    
    SimMatrix = gradientSimMatrix * PCSimMatrix * PCm
    FSIM = torch.sum(SimMatrix) / torch.sum(PCm)

    # Calculate the FSIMc
    T3 = 200
    T4 = 200
    ISimMatrix = (2 * I1 * I2 + T3) / (I1**2 + I2**2 + T3)
    QSimMatrix = (2 * Q1 * Q2 + T4) / (Q1**2 + Q2**2 + T4)
    Lambda = 0.03

    SimMatrixC = gradientSimMatrix * PCSimMatrix
    MatrixMiddle = (ISimMatrix * QSimMatrix) ** Lambda

    SimMatrixC = SimMatrixC * MatrixMiddle 
    SimMatrixC = SimMatrixC * PCm

    FSIMc = torch.sum(SimMatrixC) / torch.sum(PCm)

    return FSIMc.item()


imageOne = np.load(r'C:\Users\86177\Desktop\1020120_x_0_y_0.npy')
imageTwo = np.load(r'C:\Users\86177\Desktop\1020120_x_0_y_0_0_0.npy')
imageOne = np.stack((imageOne,) * 3, axis=-1)
imageTwo = np.stack((imageTwo,) * 3, axis=-1)
print(FeatureSIM(ToTensor()(imageOne).unsqueeze(0), ToTensor()(imageTwo).unsqueeze(0)))
