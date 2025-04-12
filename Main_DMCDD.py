# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import pickle
import matplotlib.pyplot as plt


class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# 读取数据并预处理
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
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i + window_size])
    return np.array(windows)

def create_dataset(file_path, channels, window_size, step):
    data = load_data(file_path, channels)
    data = normalize_data(data)
    windows = create_windows(data, window_size, step)
    return windows

def create_data_loader(data, batch_size=64):
    dataset = TimeSeriesDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

class ConvMLP(nn.Module):
    def __init__(self, input_channels, hidden1_dim, hidden2_dim, hidden3_dim, latent_dim, window_size):
        super(ConvMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden1_dim, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.LeakyReLU(negative_slope=0.05),
            nn.AvgPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden1_dim, hidden2_dim, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.LeakyReLU(negative_slope=0.05),
            nn.AvgPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden2_dim, hidden3_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.AvgPool1d(kernel_size=2, stride=2),
                        
            nn.Flatten()
        )
        
        self.mapping = nn.Sequential(
            nn.Linear(hidden3_dim*(window_size//8), latent_dim, bias=False),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear(latent_dim, latent_dim, bias=False),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        output = self.mapping(z)
        return output.squeeze()


class DeepMCDD:
    def __init__(self, model, feature_dim, num_classes, nu=0.05):
        self.model = model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.nu = nu
        self.centers = torch.randn(num_classes, feature_dim, requires_grad=False) 
        self.radii = torch.ones(num_classes, requires_grad=False) 
    
    def calculate_centers(self, dataloaders):
        self.model.eval()
        with torch.no_grad():
            for class_idx in range(self.num_classes):
                total_features = 0
                num_samples = 0
                for x in dataloaders[class_idx]:
                    features = self.model(x.permute(0, 2, 1).float().to(device))
                    total_features += torch.sum(features, dim=0)
                    num_samples += len(x)
                if num_samples > 0:
                    self.centers[class_idx] = total_features / num_samples
    
    def calculate_radii(self, dataloaders):
        self.model.eval()
        with torch.no_grad():
            for class_idx in range(self.num_classes):
                all_distances = []
                num_samples = 0
                for x in dataloaders[class_idx]:
                    features = self.model(x.permute(0, 2, 1).float().to(device))
                    dist = torch.sqrt(torch.sum((features - self.centers[class_idx].to(device)) ** 2, dim=1))
                    all_distances.extend(dist.detach().cpu().numpy())
                    num_samples += len(x)
                self.radii[class_idx] = torch.tensor(np.percentile(all_distances, 100 * (1 - self.nu)))
    

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            features = self.model(x.permute(0, 2, 1).float().to(device)).cpu()
            dist_to_centers = torch.cdist(features, self.centers) 
            min_dist, min_idx = torch.min(dist_to_centers, dim=1)
            is_known = min_dist <= self.radii[min_idx]  
            is_ambiguous = (dist_to_centers[:,0]>self.radii[0]) * (dist_to_centers[:,1]>self.radii[1])
            predicted_labels = torch.where(is_known, min_idx, torch.tensor(-2))
            predicted_labels = torch.where(is_ambiguous, predicted_labels, torch.tensor(-1))
            return predicted_labels, is_known.cpu().numpy() 


channels = [1,5,9,13, 20,21,22,23]
window_size = 192
hidden1_dim = 16
hidden2_dim = 24
hidden3_dim = 32
latent_dim = 1024
batch_size = 256
step = 1
nl = 0

if torch.cuda.is_available():
    print('CUDA is available. Using GPU for computation.')
    device = torch.device('cuda')
else:
    print('CUDA is not available. Using CPU for computation.')
    device = torch.device('cpu')
    torch.cuda.empty_cache()

print('loading the trained ConvMLP')
train_data_pos = create_dataset(f"BenchmarkData/{nl}/train0.csv", channels, window_size,step)
train_data_neg = create_dataset(f"BenchmarkData/{nl}/train6.csv", channels, window_size,step)
train_loader_pos = create_data_loader(train_data_pos, batch_size)
train_loader_neg = create_data_loader(train_data_neg, batch_size)
dataloaders = {0: train_loader_pos, 1: train_loader_neg}
model = ConvMLP(input_channels=len(channels), hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim, hidden3_dim=hidden3_dim, latent_dim=latent_dim, window_size=window_size).to(device)
model.load_state_dict(torch.load(f'Trained_Models_DMCDD/{nl}/ConvMLP_{window_size}.pth'))
deep_mcdd = DeepMCDD(model, latent_dim, 2)
with open(f'Trained_Models_DMCDD/{nl}/DMCDD_model_{window_size}.pkl', 'rb') as f: deep_mcdd = pickle.load(f)
    

test_data_00 = create_dataset(f"BenchmarkData/{nl}/train0.csv",channels,window_size,step)
test_data_0 = create_dataset(f"BenchmarkData/{nl}/test0.csv",channels,window_size,step)
test_data_1 = create_dataset(f"BenchmarkData/{nl}/test1.csv",channels,window_size,step)
test_data_2 = create_dataset(f"BenchmarkData/{nl}/test2.csv",channels,window_size,step)
test_data_3 = create_dataset(f"BenchmarkData/{nl}/test3.csv",channels,window_size,step)
test_data_4 = create_dataset(f"BenchmarkData/{nl}/test4.csv",channels,window_size,step)
test_data_5 = create_dataset(f"BenchmarkData/{nl}/test5.csv",channels,window_size,step)
test_data_6 = create_dataset(f"BenchmarkData/{nl}/test6.csv",channels,window_size,step)
test_data_66 = create_dataset(f"BenchmarkData/{nl}/train6.csv",channels,window_size,step)

prediction00, is_known00 = deep_mcdd.predict(torch.tensor(test_data_00))
prediction0, is_known0 = deep_mcdd.predict(torch.tensor(test_data_0))
prediction1, is_known1 = deep_mcdd.predict(torch.tensor(test_data_1))
prediction2, is_known2 = deep_mcdd.predict(torch.tensor(test_data_2))
prediction3, is_known3 = deep_mcdd.predict(torch.tensor(test_data_3))
prediction4, is_known4 = deep_mcdd.predict(torch.tensor(test_data_4))
prediction5, is_known5 = deep_mcdd.predict(torch.tensor(test_data_5))
prediction6, is_known6 = deep_mcdd.predict(torch.tensor(test_data_6))
prediction66, is_known66 = deep_mcdd.predict(torch.tensor(test_data_66))

hist0, _ = np.histogram(prediction0.numpy(), bins=[-2.5,-1.5,-0.5,0.5,1.5]); hist0 = hist0/sum(hist0)
hist1, _ = np.histogram(prediction1.numpy(), bins=[-2.5,-1.5,-0.5,0.5,1.5]); hist1 = hist1/sum(hist1)
hist2, _ = np.histogram(prediction2.numpy(), bins=[-2.5,-1.5,-0.5,0.5,1.5]); hist2 = hist2/sum(hist2)
hist3, _ = np.histogram(prediction3.numpy(), bins=[-2.5,-1.5,-0.5,0.5,1.5]); hist3 = hist3/sum(hist3)
hist4, _ = np.histogram(prediction4.numpy(), bins=[-2.5,-1.5,-0.5,0.5,1.5]); hist4 = hist4/sum(hist4)
hist5, _ = np.histogram(prediction5.numpy(), bins=[-2.5,-1.5,-0.5,0.5,1.5]); hist5 = hist5/sum(hist5)
hist6, _ = np.histogram(prediction6.numpy(), bins=[-2.5,-1.5,-0.5,0.5,1.5]); hist6 = hist6/sum(hist6)

hist_all = np.array([hist0,hist1,hist2,hist3,hist4,hist5,hist6])

heatmap = plt.imshow(hist_all, cmap='viridis')
plt.colorbar(heatmap)
plt.xticks(np.arange(4),labels=['Unk','Amb','DP0','DP6'])
plt.yticks(np.arange(7),labels=['DP0','DP1','DP2','DP3','DP4','DP5','DP6'])