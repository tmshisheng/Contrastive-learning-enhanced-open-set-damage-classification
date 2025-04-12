# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import weibull_min
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import pickle
import matplotlib.pyplot as plt

class TimeSeriesDataset(Dataset):
    def __init__(self, data, class_labels):
        self.data = data
        self.class_labels = class_labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.class_labels[idx]

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

def create_dataset(file_path, channels, window_size, step, label):
    data = load_data(file_path, channels)
    data = normalize_data(data)
    windows = create_windows(data, window_size, step)
    class_labels = np.full((windows.shape[0], ), label)
    return windows, class_labels

def create_data_loader(data, class_labels, batch_size=64):
    dataset = TimeSeriesDataset(data, class_labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

class ConvClassifier(nn.Module):
    def __init__(self, input_channels, hidden1_dim, hidden2_dim, hidden3_dim, latent_dim, window_size):
        super(ConvClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden1_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.05),
            nn.AvgPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden1_dim, hidden2_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.05),
            nn.AvgPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden2_dim, hidden3_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.05),
            nn.AvgPool1d(kernel_size=2, stride=2),
                        
            nn.Flatten()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden3_dim*((window_size)//8), (hidden3_dim*((window_size)//8) + 2)//2),
            nn.LeakyReLU(negative_slope=0.05),
            nn.Linear((hidden3_dim*((window_size)//8) + 2)//2, 2)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        classes = self.classifier(z.squeeze())
        return classes


class OpenMaxClassifier:
    def __init__(self, alpha=30, tail_size=1000):
        self.alpha = alpha
        self.tail_size = tail_size
        self.weibull_models = {}
        self.class_centers = {}
        
    def fit(self, X_train, y_train):
        self.class_centers = self.compute_class_centers(X_train, y_train)
        distances = self.compute_distances(X_train, self.class_centers)
        self.weibull_models = self.weibull_fit(distances)

    def compute_class_centers(self, X, y):
        classes = np.unique(y)
        centers = {}
        for cls in classes:
            centers[cls] = np.mean(X[y == cls], axis=0)
        return centers

    def compute_distances(self, samples, mean_vectors):
        distances = {}
        for cls, mean_vector in mean_vectors.items():
            distances[cls] = [euclidean(sample, mean_vector) for sample in samples]
        return distances

    def weibull_fit(self, distances):
        weibull_models = {}
        for cls, dists in distances.items():
            dists = sorted(dists)
            tail_dists = dists[-self.tail_size:] 
            weibull_models[cls] = weibull_min.fit(tail_dists, floc=0)
        return weibull_models

    def openmax_adjustment(self, sample):
        probabilities = {}
        for cls, mean_vector in self.class_centers.items():
            distance = euclidean(sample, mean_vector)
            w_score = weibull_min.cdf(distance, *self.weibull_models[cls])
            openmax_prob = (1 - w_score) * np.exp(-distance / self.alpha)
            probabilities[cls] = openmax_prob
        return probabilities

    def predict(self, sample, threshold=0.9):
        openmax_probs = self.openmax_adjustment(sample)
        max_prob_cls = max(openmax_probs, key=openmax_probs.get)
        max_prob = openmax_probs[max_prob_cls]
        if max_prob < threshold:
            return -1
        
        return max_prob_cls



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
    

print('loading the trained ConvClassifier')
train_data_pos,classlabels_pos = create_dataset(f"BenchmarkData/{nl}/train0.csv", channels, window_size,step,0)
train_data_neg,classlabels_neg = create_dataset(f"BenchmarkData/{nl}/train6.csv", channels, window_size,step,1)
train_loader_pos = create_data_loader(train_data_pos,classlabels_pos, batch_size)
train_loader_neg = create_data_loader(train_data_neg,classlabels_neg, batch_size)
model = ConvClassifier(input_channels=len(channels), hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim, hidden3_dim=hidden3_dim, latent_dim=latent_dim, window_size=window_size).to(device)
model.load_state_dict(torch.load(f'Trained_Models_OpenMax/{nl}/ConvAutoencoder_{window_size}.pth'))
with open(f'Trained_Models_OpenMax/{nl}/openmax_model.pkl', 'rb') as f: openmax = pickle.load(f)

# Test the OpenMax model
test_data_0,_ = create_dataset(f"BenchmarkData/{nl}/test0.csv",channels,window_size,step,0)
test_data_1,_ = create_dataset(f"BenchmarkData/{nl}/test1.csv",channels,window_size,step,0)
test_data_2,_ = create_dataset(f"BenchmarkData/{nl}/test2.csv",channels,window_size,step,0)
test_data_3,_ = create_dataset(f"BenchmarkData/{nl}/test3.csv",channels,window_size,step,0)
test_data_4,_ = create_dataset(f"BenchmarkData/{nl}/test4.csv",channels,window_size,step,0)
test_data_5,_ = create_dataset(f"BenchmarkData/{nl}/test5.csv",channels,window_size,step,0)
test_data_6,_ = create_dataset(f"BenchmarkData/{nl}/test6.csv",channels,window_size,step,0)
test_data_0 = torch.tensor(test_data_0).permute(0, 2, 1).float()
test_data_1 = torch.tensor(test_data_1).permute(0, 2, 1).float()
test_data_2 = torch.tensor(test_data_2).permute(0, 2, 1).float()
test_data_3 = torch.tensor(test_data_3).permute(0, 2, 1).float()
test_data_4 = torch.tensor(test_data_4).permute(0, 2, 1).float()
test_data_5 = torch.tensor(test_data_5).permute(0, 2, 1).float()
test_data_6 = torch.tensor(test_data_6).permute(0, 2, 1).float()
prediction0 = np.array([openmax.predict(sample) for sample in model(test_data_0.to(device)).detach().cpu().numpy()])
prediction1 = np.array([openmax.predict(sample) for sample in model(test_data_1.to(device)).detach().cpu().numpy()])
prediction2 = np.array([openmax.predict(sample) for sample in model(test_data_2.to(device)).detach().cpu().numpy()])
prediction3 = np.array([openmax.predict(sample) for sample in model(test_data_3.to(device)).detach().cpu().numpy()])
prediction4 = np.array([openmax.predict(sample) for sample in model(test_data_4.to(device)).detach().cpu().numpy()])
prediction5 = np.array([openmax.predict(sample) for sample in model(test_data_5.to(device)).detach().cpu().numpy()])
prediction6 = np.array([openmax.predict(sample) for sample in model(test_data_6.to(device)).detach().cpu().numpy()])
hist0, _ = np.histogram(prediction0, bins=[-1.5,-0.5,0.5,1.5]); hist0 = hist0/sum(hist0)
hist1, _ = np.histogram(prediction1, bins=[-1.5,-0.5,0.5,1.5]); hist1 = hist1/sum(hist1)
hist2, _ = np.histogram(prediction2, bins=[-1.5,-0.5,0.5,1.5]); hist2 = hist2/sum(hist2)
hist3, _ = np.histogram(prediction3, bins=[-1.5,-0.5,0.5,1.5]); hist3 = hist3/sum(hist3)
hist4, _ = np.histogram(prediction4, bins=[-1.5,-0.5,0.5,1.5]); hist4 = hist4/sum(hist4)
hist5, _ = np.histogram(prediction5, bins=[-1.5,-0.5,0.5,1.5]); hist5 = hist5/sum(hist5)
hist6, _ = np.histogram(prediction6, bins=[-1.5,-0.5,0.5,1.5]); hist6 = hist6/sum(hist6)
hist_all = np.array([hist0,hist1,hist2,hist3,hist4,hist5,hist6])

heatmap = plt.imshow(hist_all, cmap='viridis')
plt.colorbar(heatmap)
plt.xticks(np.arange(4),labels=['Unk','Amb','DP0','DP6'])
plt.yticks(np.arange(7),labels=['DP0','DP1','DP2','DP3','DP4','DP5','DP6'])