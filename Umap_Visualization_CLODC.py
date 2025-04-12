import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


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

# -----------------------------------------------------------------------------

def visualize_latent_space_umap(model, test_loaders):
    model.eval()
    latent_vectors_test = []
    test_labels = []
    with torch.no_grad():
        for i, test_loader in enumerate(test_loaders):
            for batch in test_loader:
                _,_,_,_,z = model(batch[0].permute(0, 2, 1).float())
                latent_vectors_test.append(z.squeeze())
                test_labels.extend([i] * len(z.squeeze()))
    
    latent_vectors_test = torch.cat(latent_vectors_test).numpy()
    
    umap_model = umap.UMAP(n_components=2)
    reduced_vectors_test = umap_model.fit_transform(latent_vectors_test)
    
    # Split the reduced vectors back into separate arrays
    plt.figure()
    scatter = plt.scatter(reduced_vectors_test[:, 0], reduced_vectors_test[:, 1], c=test_labels, cmap='Dark2', alpha=0.5)
    plt.title('Latent Space of Test Data (UMAP)')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend(*scatter.legend_elements(), title="Classes")

    plt.tight_layout()
    plt.show()
    return reduced_vectors_test, test_labels

# -----------------------------------------------------------------------------

# ConvAutoencoder模型
class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels, hidden1_dim, hidden2_dim, hidden3_dim, latent_dim, window_size):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden1_dim, kernel_size=3, stride=1, padding=1),  # [B, hidden1_dim, window_size]
            nn.LeakyReLU(negative_slope=0.05),
            nn.AvgPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden1_dim, hidden2_dim, kernel_size=3, stride=1, padding=1),  # [B, hidden2_dim, window_size/2]
            nn.LeakyReLU(negative_slope=0.05),
            nn.AvgPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden2_dim, hidden3_dim, kernel_size=3, stride=1, padding=1),  # [B, hidden2_dim, window_size/2]
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

# -----------------------------------------------------------------------------

channels = [1,5,9,13, 20,21,22,23]
window_size = 192 + 1
nl = 0
batch_size = 256
step = 10

hidden1_dim = 16
hidden2_dim = 24
hidden3_dim = 32
latent_dim = 1024

model = ConvAutoencoder(input_channels=len(channels), hidden1_dim=hidden1_dim, hidden2_dim=hidden2_dim, hidden3_dim=hidden3_dim, latent_dim=latent_dim, window_size=window_size)#.to(device)
model.load_state_dict(torch.load(f'Trained_Models_CLODC/{nl}/ConvAutoencoder_{window_size}.pth'))


train_data_0, train_labels_0 = create_dataset(f"BenchmarkData/{nl}/train0.csv", channels, window_size, step)
train_data_6, train_labels_6 = create_dataset(f"BenchmarkData/{nl}/train6.csv", channels, window_size, step)
val_data_0, val_labels_0 = create_dataset(f"BenchmarkData/{nl}/validation0.csv", channels, window_size,step)
val_data_6, val_labels_6 = create_dataset(f"BenchmarkData/{nl}/validation6.csv", channels, window_size,step)
train_loader_0 = create_data_loader(train_data_0, train_labels_0, batch_size)
train_loader_6 = create_data_loader(train_data_6, train_labels_6, batch_size)
validation_loader_0 = create_data_loader(val_data_0, val_labels_0, batch_size)
validation_loader_6 = create_data_loader(val_data_6, val_labels_6, batch_size)

test_data_0 = create_dataset(f"BenchmarkData/{nl}/test0.csv",channels,window_size)
test_data_1 = create_dataset(f"BenchmarkData/{nl}/test1.csv",channels,window_size)
test_data_2 = create_dataset(f"BenchmarkData/{nl}/test2.csv",channels,window_size)
test_data_3 = create_dataset(f"BenchmarkData/{nl}/test3.csv",channels,window_size)
test_data_4 = create_dataset(f"BenchmarkData/{nl}/test4.csv",channels,window_size)
test_data_5 = create_dataset(f"BenchmarkData/{nl}/test5.csv",channels,window_size)
test_data_6 = create_dataset(f"BenchmarkData/{nl}/test6.csv",channels,window_size)
test_loader_0 = create_data_loader(test_data_0[0],test_data_0[1],batch_size)
test_loader_1 = create_data_loader(test_data_1[0],test_data_1[1],batch_size)
test_loader_2 = create_data_loader(test_data_2[0],test_data_2[1],batch_size)
test_loader_3 = create_data_loader(test_data_3[0],test_data_3[1],batch_size)
test_loader_4 = create_data_loader(test_data_4[0],test_data_4[1],batch_size)
test_loader_5 = create_data_loader(test_data_5[0],test_data_5[1],batch_size)
test_loader_6 = create_data_loader(test_data_6[0],test_data_6[1],batch_size)

reduced_vectors_test, test_labels = visualize_latent_space_umap(model, [test_loader_0,test_loader_1,test_loader_2,test_loader_3,test_loader_4,test_loader_5,test_loader_6])
