import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class VQTrajectoryEncoder(nn.Module):
    def __init__(self, input_channels, action_dim, latent_dim, hidden_dim=128):
        super(VQTrajectoryEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc_image = nn.Linear(18432, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, img_batch, act_batch):
        batch_size, seq_len, _, _, _ = img_batch.size()
        img_batch = img_batch.view(batch_size * seq_len, img_batch.size(2), img_batch.size(3), img_batch.size(4))
        
        x = F.relu(self.conv1(img_batch))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_image(x))
        x = x.view(batch_size, seq_len, -1)
        # x = torch.cat([x, act_batch.unsqueeze(-1)], dim=2)
        lstm_out, _ = self.lstm(x)
        
        latent = self.fc(lstm_out)
        return latent
    
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z):
        z_flattened = z.view(-1, self.embedding_dim)
        distances = (torch.sum(z_flattened**2, dim=1, keepdim=True)
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(z_flattened, self.embeddings.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embeddings.weight).view(z.shape)
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        return quantized, loss, encoding_indices

class VQTrajectoryDecoder(nn.Module):
    def __init__(self, latent_dim, action_dim, output_channels, hidden_dim=128):
        super(VQTrajectoryDecoder, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc_image = nn.Linear(hidden_dim, 128 * 6 * 6)  


        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  
        self.conv4 = nn.Conv2d(16, output_channels, kernel_size=3, padding=1)

    def forward(self, quantized_latent):
        lstm_out, _ = self.lstm(quantized_latent)
        # action_preds = self.fc_action(lstm_out)

        
        x = F.relu(self.fc_image(lstm_out))
        x = x.view(x.size(0) * x.size(1), 128, 6, 6)  

        
        x = self.upsample1(x)
        x = F.relu(self.conv1(x))  

        x = self.upsample2(x)
        x = F.relu(self.conv2(x))  

        x = self.upsample3(x)
        x = F.relu(self.conv3(x))  

        x = self.upsample4(x)
        recon_images = torch.sigmoid(self.conv4(x)) 
        recon_images = recon_images.view(lstm_out.size(0), lstm_out.size(1), *recon_images.shape[1:])
        
        return recon_images, {}


class VQ_TrajectoryVAE(nn.Module):
    def __init__(self, input_channels, action_dim, latent_dim, num_embeddings, output_channels, hidden_dim=128):
        super(VQ_TrajectoryVAE, self).__init__()
        self.encoder = VQTrajectoryEncoder(input_channels, action_dim, latent_dim, hidden_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = VQTrajectoryDecoder(latent_dim, action_dim, output_channels, hidden_dim)

    def forward(self, img_batch, act_batch):
        latent = self.encoder(img_batch, act_batch)
        quantized, vq_loss, encoding_indices = self.vq_layer(latent)
        recon_images, action_preds = self.decoder(quantized)
        return recon_images, action_preds, vq_loss, encoding_indices