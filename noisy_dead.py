import os, torch, time
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from math import e
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict

n_epochs = 100
p_thresh = 22
s_thresh = 0.7

class Noisy:
    def __init__(self, name, mu, iteration, psnr, ssim, dead):
        self.name = name
        self.mu = mu
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
layers = 5
neurons = 256
sidelength = 128
activation_func = nn.ReLU()
k = 5
img_list = ["petal_1.png", "petal_2.png", "petal_3.png", "cameraman.png"]
mu_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# img_list = ["petal_1.png"]
# mu_list = [0]

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


### TRAINING ###

all_stats = []

start = time.time()
for img_path in img_list:
    img_name = img_path.replace(".png", "")
    img = Image.open(img_path).resize((sidelength, sidelength))
    img = torch.tensor(np.array(img)).float() / 255.0
    img = img.reshape(-1, 1).to(device)

    for mu in mu_list:
        img_noisy = noise_function(mu, img.clone(), torch.randn_like(img))
        img_noisy = img_noisy.reshape(-1, 1).to(device)
        
        for iteration in range(10):
            print(f"Running {img_name} | mu {mu} | iteration {iteration + 1} of 10")

            model = MLP().to(device)
            loss_func = nn.MSELoss()
            opt = Adam(model.parameters(), lr=learning_rate)
            dataset = IMGDataset(xy, img_noisy)
            dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

            save_path1 = f'noise_dead1/{img_name}/{mu}/PSNR'
            save_path2 = f'noise_dead1/{img_name}/{mu}/IMG'
            save_path3 = f'noise_dead1/{img_name}/{mu}/Feature-maps'
            os.makedirs(save_path1, exist_ok=True)
            os.makedirs(save_path2, exist_ok=True)
            os.makedirs(save_path3, exist_ok=True)

            losses, accuracies, accuracies_SSIM = [], [], []
            for epoch in range(n_epochs):
                epoch_losses = []
                for batch in dataloader:
                    x, y = batch
                    batch_loss = train_batch(x, y, model, opt, loss_func)
                    epoch_losses.append(batch_loss)

                losses.append(np.mean(epoch_losses))
                accuracies.append(accuracy_PSNR(xy, img_noisy, model))
                accuracies_SSIM.append(accuracy_SSIM(xy, img_noisy, model))

            epochs = np.arange(n_epochs) + 1
            plt.figure(figsize=(30, 3))

            plt.subplot(131)
            plt.title(f'Training Loss over {n_epochs} epochs')
            plt.plot(epochs, losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            plt.subplot(132)
            plt.title(f'Training PSNR over {n_epochs} epochs')
            plt.plot(epochs, accuracies)
            plt.xlabel('Epoch')
            plt.ylabel('PSNR')

            plt.subplot(133)
            plt.title(f'Training SSIM over {n_epochs} epochs')
            plt.plot(epochs, accuracies_SSIM)
            plt.xlabel('Epoch')
            plt.ylabel('SSIM')

            plt.tight_layout()
            plt.savefig(os.path.join(save_path1, f"PSNR_{iteration}.png"))
            plt.close()

            model.eval()
            with torch.no_grad():
                pred_colors = model(xy)
            pred_img = pred_colors.view(sidelength, sidelength).cpu().numpy()
            noisy_img = img_noisy.view(sidelength, sidelength).cpu().numpy()

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(noisy_img, cmap='gray')
            plt.title(f"Noisy Input from {mu}")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(pred_img, cmap='gray')
            plt.title(f"Denoised Output from {mu}")
            plt.axis("off")
            plt.savefig(os.path.join(save_path2, f"img_{iteration}.png"))
            plt.close()

            cluster_maps = []
            dead_neurons = {}

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

            for layer_name, features in layer_features.items():
                threshold = 0.2
                total_non_activated = 0
                completely_inactive_features = 0
                num_samples, num_features = features.shape

                for feature_idx in range(num_features):
                    feature_map = features[:, feature_idx].reshape(sidelength, sidelength)
                    feature_normalized = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                    non_activated_pixels = np.sum(feature_normalized < threshold)
                    total_non_activated += non_activated_pixels
                    if non_activated_pixels == (sidelength * sidelength):
                        completely_inactive_features += 1

                total_pixels = num_features * sidelength * sidelength
                inactive_features_percentage = (completely_inactive_features / num_features) * 100
                non_activated_percentage = (total_non_activated / total_pixels) * 100
                dead_neurons[f"{img_name}_L{layer_name.split('_')[-1]}"] = inactive_features_percentage

                feature_maps = features.T.reshape(num_features, sidelength, sidelength)
                channels_per_plot = 20
                cols = 5
                rows = 4
                num_plots = (num_features + channels_per_plot - 1) // channels_per_plot

                for plot_num in range(num_plots):
                    start_ch = plot_num * channels_per_plot
                    end_ch = min(start_ch + channels_per_plot, num_features)
                    plt.figure(figsize=(15, 12))
                    plt.suptitle(f'{layer_name.title()}  - Plot{plot_num + 1} - Features {start_ch} to {end_ch-1}\nNon-activated: {non_activated_percentage:.1f}% | Inactive: {inactive_features_percentage:.1f}%')

                    for idx, ch in enumerate(range(start_ch, end_ch)):
                        plt.subplot(rows, cols, idx + 1)
                        im = plt.imshow(feature_maps[ch], cmap='viridis')
                        plt.title(f'Feature {ch}', fontsize=10)
                        plt.colorbar(im, shrink=0.6)
                        plt.axis('off')

                    filename = f"{layer_name}_plot{plot_num + 1}_features_{start_ch}-{end_ch-1}_{iteration}.png"
                    plt.savefig(os.path.join(save_path3, filename), dpi=150, bbox_inches='tight')
                    plt.close()

                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(features)
                cluster_map = cluster_labels.reshape(sidelength, sidelength)
                cluster_maps.append((cluster_map, f'{layer_name} - K-means (k={k})'))

            cols = 3
            rows = int(np.ceil(len(cluster_maps) / cols))
            plt.figure(figsize=(6 * cols, 5 * rows))
            for i, (cluster_map, title) in enumerate(cluster_maps):
                plt.subplot(rows, cols, i + 1)
                im = plt.imshow(cluster_map, cmap='tab10')
                plt.title(title, fontsize=12)
                plt.axis("off")
                plt.colorbar(im, shrink=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path3, f"cluster-maps_{iteration}.png"), dpi=150, bbox_inches='tight')
            plt.close()

            psnr_val = accuracy_PSNR(xy, img_noisy, model)
            ssim_val = accuracy_SSIM(xy, img_noisy, model)
            entry = Noisy(img_name, mu, iteration, psnr_val, ssim_val, dead_neurons)
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
        "iteration": stat.get_iteration(),
        "psnr": round(stat.get_psnr(), 2),
        "ssim": round(stat.get_ssim(), 2)
    }
    for layer_key, val in stat.get_dead().items():
        short_layer = "L" + layer_key.split('_')[-1]
        row[short_layer] = round(val, 2)
    rows.append(row)

df_stats = pd.DataFrame(rows)
df_stats.to_csv("noise_dead1/all_stats.csv", index=False)

grouped = defaultdict(list)
for stat in all_stats:
    key = (stat.get_name(), stat.get_mu())
    grouped[key].append(stat)

avg_rows = []

for (image, mu), stats in grouped.items():
    filtered = [s for s in stats if s.psnr_above_thresh(p_thresh) and s.ssim_above_thresh(s_thresh)]

    excluded_count = len(stats) - len(filtered)
    if not filtered:
        filtered = stats
        excluded_count = 0

    row = {"image": image, "mu": mu}
    layer_data = defaultdict(list)
    for stat in filtered:
        for layer_key, val in stat.get_dead().items():
            short_layer = "L" + layer_key.split('_')[-1]
            layer_data[short_layer].append(val)
    for layer, vals in layer_data.items():
        row[layer] = round(np.mean(vals), 2)
    
    row["excluded"] = excluded_count
    avg_rows.append(row)

df_avg = pd.DataFrame(avg_rows)
df_avg = df_avg.sort_values(by=["image", "mu"]).reset_index(drop=True)
df_avg.to_csv("noise_dead1/avg_dead.csv", index=False)


