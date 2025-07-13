import os, torch, time
import numpy as np
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from kneed import KneeLocator

class Noisy:
    def __init__(self, name, mu, pattern, iteration, psnr, ssim, dead):
        self.name = name
        self.mu = mu
        self.pattern = pattern
        self.iteration = iteration
        self.psnr = psnr
        self.ssim = ssim
        self.dead = dead
    
    def psnr_above_thresh(self, thresh):
        return self.psnr >= thresh

    def ssim_above_thresh(self, thresh):
        return self.ssim >= thresh
    
    def get_name(self):
        return self.name
    
    def get_mu(self):
        return self.mu

    def get_pattern(self):
        return self.pattern

    def get_iteration(self):
        return self.iteration

    def get_psnr(self):
        return self.psnr
    
    def get_ssim(self):
        return self.ssim
    
    def get_dead(self):
        return self.dead

def noise_function(mu, I, N):
    noisy = (1 - mu) * I + mu * N
    noisy = noisy - torch.min(noisy)
    noisy = noisy / torch.max(noisy)
    return noisy

learning_rate = 1e-3
n_epochs = 100
layers = 5
neurons = 256
sidelength = 128
activation_func = nn.ReLU()
img_list = ["petal_1.png", "petal_2.png", "petal_3.png", "cameraman.png"]
mu_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

x = np.arange(0, sidelength) - (sidelength - 1) / 2
y = np.arange(0, sidelength) - (sidelength - 1) / 2
x, y = np.meshgrid(x, y, indexing='xy')
x = x / ((sidelength - 1) / 2)
y = y / ((sidelength - 1) / 2)
xy = np.stack([x, y], axis=2)
xy = torch.Tensor(xy).float().reshape(-1, 2).to(device)

class IMGDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x.float(), y.float()
    def __getitem__(self, ix):
        return self.x[ix].to(device), self.y[ix].to(device)
    def __len__(self):
        return len(self.x)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, neurons),
            activation_func,
            nn.Linear(neurons, neurons),
            activation_func,
            nn.Linear(neurons, neurons),
            activation_func,
            nn.Linear(neurons, neurons),
            activation_func,
            nn.Linear(neurons, neurons),
            activation_func,
            nn.Linear(neurons, 1)
        )

    def forward(self, x):
        return self.model(x)

def train_batch(x, y, model, opt, loss_fn):
    model.train()
    opt.zero_grad()
    batch_loss = loss_fn(model(x), y)
    batch_loss.backward()
    opt.step()
    return batch_loss.detach().cpu().numpy()

def accuracy_SSIM(x, y, model):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        pred_img = prediction.view(sidelength, sidelength).cpu().numpy()
        true_img = y.view(sidelength, sidelength).cpu().numpy()
        ssim_value = ssim(true_img, pred_img, data_range=true_img.max() - true_img.min())
    return ssim_value

def accuracy_PSNR(x, y, model):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        mse = torch.mean((prediction - y) ** 2)
        max_val = 1.0
        psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)
    return psnr.detach().cpu().numpy()

def save_epoch_stats(accuracies, save_path, prefix, SSIM=False):
    network_id = prefix
    new_data = pd.DataFrame({
        'Epoch': np.arange(1, n_epochs + 1),
        f'{network_id}': accuracies
    })

    csv_path = os.path.join(save_path, f"SSIM.csv" if SSIM else f"PSNR.csv")

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        combined_df = pd.merge(existing_df, new_data, on='Epoch', how='outer')
    else:
        combined_df = new_data

    combined_df.to_csv(csv_path, index=False)


def calculate_elbow_point(features, max_k=12, layer_name=""):
    unique_patterns = len(np.unique(features, axis=0))

    if unique_patterns <= 1:
        print(f"{layer_name}: Only 1 unique pattern, using k=1")
        return 1

    max_k = min(max_k, unique_patterns)
    k_range = range(2, max_k + 1)
    wcss_values = []

    print(f"Calculating elbow point for {layer_name}...")

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        cluster_labels = kmeans.fit_predict(features)
        wcss = kmeans.inertia_
        wcss_values.append(wcss)
        print(f"k={k}: WCSS={wcss:.2f}")

    knee_locator = KneeLocator(k_range, wcss_values, curve='convex', direction='decreasing')
    optimal_k = knee_locator.knee

    if optimal_k is None:
        optimal_k = min(3, unique_patterns)
        print(f"Elbow detection failed, using k={optimal_k}")
    else:
        print(f"Optimal k found: {optimal_k}")

    return optimal_k

def visualize_cluster_representatives(features, cluster_labels, cluster_centers, layer_name, sidelength, col_num):
    optimal_k = len(cluster_centers)
    for cluster_id in range(optimal_k):
        neurons_in_cluster = np.where(cluster_labels == cluster_id)[0]
        
            
        print(f"Cluster {cluster_id}: {len(neurons_in_cluster)} neurons - {neurons_in_cluster}")
        cluster_center = cluster_centers[cluster_id]
        neuron_distances = []

        for neuron_idx in neurons_in_cluster:
            neuron_pattern = features[:, neuron_idx]
            distance = np.linalg.norm(neuron_pattern - cluster_center)
            neuron_distances.append((neuron_idx, distance))

        neuron_distances.sort(key=lambda x: x[1])
        top = neuron_distances[0]
        neuron_2d = features[:, top[0]].reshape(sidelength, sidelength)

        ax = plt.subplot2grid((optimal_k, 6), (cluster_id, col_num-1), fig=plt.gcf())
        im = ax.imshow(neuron_2d, cmap='viridis')
        ax.set_title(f'N {top[0]}, D {top[1]:.3f}', fontsize=12, pad=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.6)

def perform_neuron_clustering(features, layer_name, sidelength, col_num):
    optimal_k = calculate_elbow_point(features.T, max_k=12, layer_name=f"{layer_name}_neurons")
    kmeans_neurons = KMeans(n_clusters=optimal_k, random_state=42)
    neuron_cluster_labels = kmeans_neurons.fit_predict(features.T)
    visualize_cluster_representatives(features, neuron_cluster_labels, kmeans_neurons.cluster_centers_, layer_name, sidelength, col_num)

### TRAINING ###

all_stats = []

start = time.time()
for img_path in img_list:
    img_name = img_path.replace(".png", "")
    img = Image.open(img_path).resize((sidelength, sidelength))
    img = torch.tensor(np.array(img)).float() / 255.0
    img = img.reshape(-1, 1).to(device)

    for mu in mu_list:
        for pattern in range(10):
            img_noisy = noise_function(mu, img.clone(), torch.randn_like(img))
            img_noisy = img_noisy.reshape(-1, 1).to(device)

            for iteration in range(10):
                print(f"Running {img_name} | mu {mu} | pattern {pattern} | iteration {iteration + 1} of 10")

                model = MLP().to(device)
                loss_func = nn.MSELoss()
                opt = Adam(model.parameters(), lr=learning_rate)
                dataset = IMGDataset(xy, img_noisy)
                dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

                save_path1 = f'noise_dead/{img_name}/{mu}/pattern{pattern}/PSNR'
                save_path_row = f'noise_dead/{img_name}/{mu}/pattern{pattern}/ROW'
                os.makedirs(save_path1, exist_ok=True)
                os.makedirs(save_path_row, exist_ok=True)

                losses, accuracies, accuracies_SSIM = [], [], []
                for epoch in range(n_epochs):
                    epoch_losses = []
                    for batch in dataloader:
                        x, y = batch
                        batch_loss = train_batch(x, y, model, opt, loss_func)
                        epoch_losses.append(batch_loss)

                    losses.append(np.mean(epoch_losses))
                    accuracies.append(accuracy_PSNR(xy, img, model))
                    accuracies_SSIM.append(accuracy_SSIM(xy, img, model))

                save_epoch_stats(accuracies, save_path1, f"{iteration}", SSIM=False)
                save_epoch_stats(accuracies_SSIM, save_path1, f"{iteration}", SSIM=True)

                # Extract layer features
                model.eval()
                with torch.no_grad():
                    layer_features = {}
                    x = xy
                    x = model.model[0](x)
                    x = model.model[1](x)
                    layer_features['layer_1'] = x.cpu().numpy()
                    x = model.model[2](x)
                    x = model.model[3](x)
                    layer_features['layer_2'] = x.cpu().numpy()
                    x = model.model[4](x)
                    x = model.model[5](x)
                    layer_features['layer_3'] = x.cpu().numpy()
                    x = model.model[6](x)
                    x = model.model[7](x)
                    layer_features['layer_4'] = x.cpu().numpy()
                    x = model.model[8](x)
                    x = model.model[9](x)
                    layer_features['layer_5'] = x.cpu().numpy()

                # Generate prediction
                model.eval()
                with torch.no_grad():
                    pred_colors = model(xy)
                pred_img = pred_colors.view(sidelength, sidelength).cpu().numpy()

                # Create visualization plots
                plt.figure(figsize=(30, 20))
                plt.subplot(1, 6, 1)
                plt.imshow(pred_img, cmap='gray')
                plt.title(f"Predicted Image from {iteration}\nPSNR: {accuracies[-1]:.6f}, SSIM: {accuracies_SSIM[-1]:.6f}")
                plt.axis("off")

                dead_neurons = 0
                total_neurons = 0
                dead_neurons_dict = {}

                for idx, (layer_name, features) in enumerate(layer_features.items()):
                    print(f"Processing {layer_name} with shape {features.shape}")

                    threshold = 0.5
                    completely_inactive_features = 0
                    num_features = features.shape[1]
                    total_neurons += num_features

                    for feature_idx in range(num_features):
                        feature_map = features[:, feature_idx].reshape(sidelength, sidelength)
                        feature_normalized = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                        non_activated_pixels = np.sum(feature_normalized < threshold)

                        if non_activated_pixels == (sidelength * sidelength):
                            completely_inactive_features += 1
                            dead_neurons += 1

                    inactive_features_percentage = (completely_inactive_features / num_features) * 100
                    dead_neurons_dict[f"{img_name}_L{layer_name.split('_')[-1]}"] = inactive_features_percentage

                    plt.subplot(1, 6, idx + 2)
                    plt.title(f'{layer_name.title()} - IF: {inactive_features_percentage:.1f}% (TH {threshold})', fontsize=14, pad=40)
                    plt.axis('off')
                    
                    # Perform neuron clustering
                    perform_neuron_clustering(features, layer_name, sidelength, idx + 2)

                plt.subplots_adjust(left=0.03, right=0.97, top=0.75, hspace=0.3)
                plt.savefig(os.path.join(save_path_row, f"neuron_clustering_{iteration}.png"), 
                            bbox_inches='tight', facecolor='white', edgecolor='white', pad_inches=1.5)
                plt.close()

                psnr_val = np.mean(accuracies[-5:]) if len(accuracies) >= 5 else np.mean(accuracies)
                ssim_val = np.mean(accuracies_SSIM[-5:]) if len(accuracies_SSIM) >= 5 else np.mean(accuracies_SSIM)
                entry = Noisy(img_name, mu, pattern, iteration, psnr_val, ssim_val, dead_neurons_dict)
                all_stats.append(entry)

                mid = time.time()
                print(f"Time so far: {mid - start:.2f} seconds")

end = time.time()
print(f"Training time: {end - start:.2f} seconds")

### CSV SAVE ###

# all stats
rows = []
for stat in all_stats:
    row = {
        "name": stat.get_name(),
        "mu": stat.get_mu(),
        "pattern": stat.get_pattern(),
        "iteration": stat.get_iteration(),
        "psnr": round(stat.get_psnr(), 2),
        "ssim": round(stat.get_ssim(), 2)
    }
    for layer_key, val in stat.get_dead().items():
        short_layer = "L" + layer_key.split('_')[-1]
        row[short_layer] = round(val, 2)
    rows.append(row)

df_stats = pd.DataFrame(rows)
df_stats.to_csv("noise_dead/all_stats.csv", index=False)

grouped = defaultdict(list)
for stat in all_stats:
    key = (stat.get_name(), stat.get_mu(), stat.get_pattern())
    grouped[key].append(stat)

avg_rows = []

for (image, mu, pattern), stats in grouped.items():
    row = {"image": image, "mu": mu, "pattern": pattern}
    layer_data = defaultdict(list)
    for stat in stats:
        for layer_key, val in stat.get_dead().items():
            short_layer = "L" + layer_key.split('_')[-1]
            layer_data[short_layer].append(val)
    for layer, vals in layer_data.items():
        row[layer] = round(np.mean(vals), 2)
    avg_rows.append(row)

df_avg = pd.DataFrame(avg_rows)
df_avg = df_avg.sort_values(by=["image", "mu", "pattern"]).reset_index(drop=True)
df_avg.to_csv("noise_dead/avg_dead.csv", index=False)

### HEATMAPS ###
heatmap_path = "noise_dead/heatmaps"
os.makedirs(heatmap_path, exist_ok=True)

sns.set(style="whitegrid")
layer_cols = ['L1', 'L2', 'L3', 'L4', 'L5']
images = df_avg['image'].unique()

for image in images:
    subset = df_avg[df_avg['image'] == image]
    subset = subset.sort_values(by="mu")
    
    heatmap_data = subset[layer_cols].T
    heatmap_data.columns = subset['mu'].values

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap="viridis", fmt=".2f")
    plt.title(f"Heatmap of Dead Neurons per Layer - Image: {image}")
    plt.xlabel("mu")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_path, f"{image}_heatmap.png"))
    plt.close()
