import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(EncoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # self.layer_norm = nn.LayerNorm(hidden_dim)  

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        # outputs_ln = self.layer_norm(outputs)
        
        return hidden, cell

class DecoderLSTM(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=2):
        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.layer_norm = nn.LayerNorm(hidden_dim)  

    def forward(self, x, hidden, cell):
        lstm_out, (hidden, cell) = self.lstm(x, (hidden, cell))
        # lstm_out_ln = self.layer_norm(lstm_out)
        
        output = self.fc(lstm_out)
        return output, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        hidden, cell = self.encoder(src)
        outputs, hidden, cell = self.decoder(trg, hidden, cell)
        return outputs
    


class VQEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, hidden_dim=128):
        super(VQEncoder, self).__init__()
        self.lstm = nn.LSTM(state_dim + action_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, traj):
        lstm_out, _ = self.lstm(traj) 
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

class VQDecoder(nn.Module):
    def __init__(self, latent_dim, state_dim, action_dim, hidden_dim=128):
        super(VQDecoder, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, state_dim + action_dim)

    def forward(self, quantized_latent):
        lstm_out, _ = self.lstm(quantized_latent)
        recon_traj = self.fc(lstm_out)
        return recon_traj

class VQ_VAE_Segment(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, num_embeddings):
        super(VQ_VAE_Segment, self).__init__()
        self.encoder = VQEncoder(state_dim, action_dim, latent_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = VQDecoder(latent_dim, state_dim, action_dim)

    def forward(self, traj):
        latent = self.encoder(traj)
        quantized, vq_loss, encoding_indices = self.vq_layer(latent)
        recon_traj = self.decoder(quantized)
        return recon_traj, vq_loss, encoding_indices