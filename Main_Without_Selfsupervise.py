# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 20:58:19 2024

@author: Shi Sheng

2443177 parameters

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import joblib
import matplotlib.pyplot as plt


def load_data(file_path, channels):
    df = pd.read_csv(file_path)
    data = df.iloc[:, channels].values
    return data

def normalize_data(data):
    rms = max(np.sqrt(np.mean(data**2, axis=0)))
    normalized_data = data / rms
    return normalized_data

def create_windows(data, window_size, step):
    windows = []
    labels = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i + window_size - 1])
        labels.append(data[i + window_size - 1])
    return np.array(windows), np.array(labels)

class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def create_dataset(file_path, channels, window_size, step=5):
    data = load_data(file_path, channels)
    data = normalize_data(data)
    windows, labels = create_windows(data, window_size, step)
    return windows, labels

def create_data_loader(data, labels, batch_size=64):
    dataset = TimeSeriesDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


class HyperSphereLoss(nn.Module):
    def __init__(self, lambda_val=0.05, alpha=10, nu=0.05):
        super(HyperSphereLoss, self).__init__()
        self.lambda_val = lambda_val
        self.alpha = alpha 
        self.nu = nu
        self.c1 = 0
        self.c2 = 0
        self.R1 = 0
        self.R2 = 0
        
    def forward(self, z1, z2):
        loss1 = torch.mean(torch.sum((z1 - self.c1) ** 2, dim=1))
        loss2 = torch.mean(torch.sum((z2 - self.c2) ** 2, dim=1))
        inter_class_loss = nn.functional.relu(self.R1+self.R2+self.lambda_val - torch.norm(self.c1 - self.c2))
        loss = self.alpha * inter_class_loss + (loss1+loss2)
        return loss    
        
    def predict(self, z):
        dist1 = torch.sqrt(torch.sum((z - self.c1) ** 2, dim=1))
        dist2 = torch.sqrt(torch.sum((z - self.c2) ** 2, dim=1))
        predicted_class = torch.zeros(z.size(0))
        predicted_class[(dist1 <= self.R1) & (dist2 > self.R2)] = 1
        predicted_class[(dist1 > self.R1) & (dist2 <= self.R2)] = 2
        predicted_class[(dist1 > self.R1) & (dist2 > self.R2)] = -1
        return predicted_class


class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels, hidden1_dim, hidden2_dim, hidden3_dim, latent_dim, window_size):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden1_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.AvgPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden1_dim, hidden2_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.AvgPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden2_dim, hidden3_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.05),
            nn.AvgPool1d(kernel_size=2, stride=2),
                        
            nn.Flatten()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (hidden3_dim, (window_size-1) // 8)),
            nn.Upsample(scale_factor=2), 
            nn.LeakyReLU(negative_slope=0.05),
            nn.ConvTranspose1d(hidden3_dim, hidden2_dim, kernel_size=3, stride=1, padding=1, output_padding=0),  
            nn.Upsample(scale_factor=2),  
            nn.LeakyReLU(negative_slope=0.05),
            nn.ConvTranspose1d(hidden2_dim, hidden1_dim, kernel_size=3, stride=1, padding=1, output_padding=0), 
            
            nn.Upsample(scale_factor=2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.ConvTranspose1d(hidden1_dim, input_channels, kernel_size=3, stride=1, padding=1, output_padding=0)
        )

        # Projector
        self.mapping = nn.Sequential(
            nn.Linear(hidden3_dim*((window_size-1)//8), latent_dim),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(latent_dim, latent_dim),
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden3_dim*((window_size-1)//8), (hidden3_dim*((window_size-1)//8) + input_channels)//2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear((hidden3_dim*((window_size-1)//8) + input_channels)//2, input_channels)
            )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden3_dim*((window_size-1)//8), (hidden3_dim*((window_size-1)//8) + 2)//2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear((hidden3_dim*((window_size-1)//8) + 2)//2, 2)
            )
        
    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        z_mapped = self.mapping(z)
        prediction = self.predictor(z.squeeze())
        classes = self.classifier(z.squeeze())
        return x_reconstructed, z.squeeze(), prediction.squeeze(), classes, z_mapped.squeeze()
                

def classify_and_detect_unknown(hplossfunc, model, test_loader):
    model.eval()
    latent_test = []
    with torch.no_grad():
        for batch in test_loader:
            batch,_ = batch
            batch = batch.permute(0, 2, 1).float().to(device)
            _,_,_,_,z = model(batch)
            latent_test.append(z)
    latent_test = torch.cat(latent_test)
    predicted_labels = hplossfunc.predict(latent_test)
    return predicted_labels
    
#=====================================================================================================================================

# 训练和评估
channels = [1,5,9,13, 20,21,22,23]
window_size = 192  +1
hidden1_dim = 16
hidden2_dim = 24
hidden3_dim = 32
latent_dim = 1024
batch_size = 4096
step = 1
nl = 0

if torch.cuda.is_available():
    print('CUDA is available. Using GPU for computation.')
    device = torch.device('cuda')
else:
    print('CUDA is not available. Using CPU for computation.')
    device = torch.device('cpu')
    torch.cuda.empty_cache()


print('Loading the trained ConvAutoencoder')
model = ConvAutoencoder(input_channels=len(channels), hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim, hidden3_dim=hidden3_dim, latent_dim=latent_dim, window_size=window_size).to(device)
model.load_state_dict(torch.load(f'Trained_Models_Without_Selfsupervise/{nl}/ConvAutoencoder_{window_size}.pth'))
hplossfunc = joblib.load(f'Trained_Models_Without_Selfsupervise/{nl}/hplossfunc_{window_size}.pth')


print('Start testing')
# 加载测试数据
test_data_0  = create_dataset(f"BenchmarkData/{nl}/test0.csv" ,channels,window_size,1)
test_data_1  = create_dataset(f"BenchmarkData/{nl}/test1.csv" ,channels,window_size,1)
test_data_2  = create_dataset(f"BenchmarkData/{nl}/test2.csv" ,channels,window_size,1)
test_data_3  = create_dataset(f"BenchmarkData/{nl}/test3.csv" ,channels,window_size,1)
test_data_4  = create_dataset(f"BenchmarkData/{nl}/test4.csv" ,channels,window_size,1)
test_data_5  = create_dataset(f"BenchmarkData/{nl}/test5.csv" ,channels,window_size,1)
test_data_6  = create_dataset(f"BenchmarkData/{nl}/test6.csv" ,channels,window_size,1)

test_loader_0 = create_data_loader(test_data_0[0],test_data_0[1],batch_size)
test_loader_1 = create_data_loader(test_data_1[0],test_data_1[1],batch_size)
test_loader_2 = create_data_loader(test_data_2[0],test_data_2[1],batch_size)
test_loader_3 = create_data_loader(test_data_3[0],test_data_3[1],batch_size)
test_loader_4 = create_data_loader(test_data_4[0],test_data_4[1],batch_size)
test_loader_5 = create_data_loader(test_data_5[0],test_data_5[1],batch_size)
test_loader_6 = create_data_loader(test_data_6[0],test_data_6[1],batch_size)

# 开集识别
predicted_labels_0 = classify_and_detect_unknown(hplossfunc, model, test_loader_0).detach().numpy()
predicted_labels_1 = classify_and_detect_unknown(hplossfunc, model, test_loader_1).detach().numpy()
predicted_labels_2 = classify_and_detect_unknown(hplossfunc, model, test_loader_2).detach().numpy()
predicted_labels_3 = classify_and_detect_unknown(hplossfunc, model, test_loader_3).detach().numpy()
predicted_labels_4 = classify_and_detect_unknown(hplossfunc, model, test_loader_4).detach().numpy()
predicted_labels_5 = classify_and_detect_unknown(hplossfunc, model, test_loader_5).detach().numpy()
predicted_labels_6 = classify_and_detect_unknown(hplossfunc, model, test_loader_6).detach().numpy()

hist0, _ = np.histogram(predicted_labels_0, bins=[-1.5,-0.5,0.5,1.5,2.5]); hist0 = hist0/sum(hist0)
hist1, _ = np.histogram(predicted_labels_1, bins=[-1.5,-0.5,0.5,1.5,2.5]); hist1 = hist1/sum(hist1)
hist2, _ = np.histogram(predicted_labels_2, bins=[-1.5,-0.5,0.5,1.5,2.5]); hist2 = hist2/sum(hist2)
hist3, _ = np.histogram(predicted_labels_3, bins=[-1.5,-0.5,0.5,1.5,2.5]); hist3 = hist3/sum(hist3)
hist4, _ = np.histogram(predicted_labels_4, bins=[-1.5,-0.5,0.5,1.5,2.5]); hist4 = hist4/sum(hist4)
hist5, _ = np.histogram(predicted_labels_5, bins=[-1.5,-0.5,0.5,1.5,2.5]); hist5 = hist5/sum(hist5)
hist6, _ = np.histogram(predicted_labels_6, bins=[-1.5,-0.5,0.5,1.5,2.5]); hist6 = hist6/sum(hist6)

hist_all = np.array([hist0,hist1,hist2,hist3,hist4,hist5,hist6])

heatmap = plt.imshow(hist_all, cmap='viridis')
plt.colorbar(heatmap)
plt.xticks(np.arange(4),labels=['Unk','Amb','DP0','DP6'])
plt.yticks(np.arange(7),labels=['DP0','DP1','DP2','DP3','DP4','DP5','DP6'])