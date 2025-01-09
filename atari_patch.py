import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, image_size):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.n_patches = self.grid_size * self.grid_size
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
    def forward(self, x):
        B, C, H, W = x.shape
        p = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        p = p.reshape(B, C, -1, self.patch_size, self.patch_size).permute(0, 2, 1, 3, 4)
        p = p.reshape(B, self.n_patches, -1)
        return self.proj(p)

class PatchReconstruction(nn.Module):
    def __init__(self, out_channels, patch_size, embed_dim, image_size):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.n_patches = self.grid_size * self.grid_size
        self.reconstruct = nn.Linear(embed_dim, out_channels * patch_size * patch_size)
    def forward(self, x):
        B, N, D = x.shape
        p = self.reconstruct(x).view(B, N, -1, self.patch_size, self.patch_size)
        p = p.permute(0, 2, 1, 3, 4).contiguous()
        p = p.view(B, -1, self.grid_size, self.grid_size, self.patch_size, self.patch_size)
        p = p.permute(0, 1, 2, 4, 3, 5).contiguous()
        return p.view(B, -1, self.grid_size * self.patch_size, self.grid_size * self.patch_size)

class PatchAggregator(nn.Module):
    def __init__(self, embed_dim, n_heads=4, num_layers=2, ff=256):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        layer = nn.TransformerEncoderLayer(embed_dim, n_heads, ff, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers)
    def forward(self, patches):
        B, N, D = patches.shape
        cls_token = self.cls.expand(B, -1, -1)
        x = torch.cat([cls_token, patches], dim=1)
        out = self.enc(x)
        return out[:, 0, :]

class VectorQuantizer(nn.Module):
    def __init__(self, num_tokens, embed_dim, beta=1.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.beta = beta
        self.codebook = nn.Embedding(num_tokens, embed_dim)
        nn.init.uniform_(self.codebook.weight, -1.0/num_tokens, 1.0/num_tokens)
    def forward(self, z):
        B, T, D = z.shape
        f = z.view(-1, D)
        dist = (f**2).sum(dim=1, keepdim=True) - 2*f@self.codebook.weight.T + (self.codebook.weight**2).sum(dim=1)
        idx = dist.argmin(dim=-1)
        z_q = self.codebook(idx).view(B, T, D)
        c_loss = self.beta*F.mse_loss(z.detach(), z_q)
        cb_loss = F.mse_loss(z, z_q.detach())
        z_q = z + (z_q - z).detach()
        return z_q, c_loss, cb_loss

class SeqEncoder(nn.Module):
    def __init__(self, embed_dim, n_heads=8, num_layers=4, ff=512):
        super().__init__()
        layer = nn.TransformerEncoderLayer(embed_dim, n_heads, ff, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers)
    def forward(self, x):
        return self.enc(x)

class FrameDecoder(nn.Module):
    def __init__(self, embed_dim, patch_size, image_size):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.out = PatchReconstruction(1, patch_size, embed_dim, image_size)
    def forward(self, z):
        return self.out(z)

class SeqDecoder(nn.Module):
    def __init__(self, embed_dim, n_heads=8, num_layers=4, ff=512):
        super().__init__()
        layer = nn.TransformerDecoderLayer(embed_dim, n_heads, ff, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers)
    def mask(self, L):
        return torch.triu(torch.ones(L, L)*float('-inf'), diagonal=1)
    def forward_autoregressive(self, memory, teacher_forcing_ratio, gt_frames):
        B, T, D = memory.shape
        preds = []
        dec_in = torch.zeros(B, 0, D, device=memory.device)
        for t in range(T):
            L = dec_in.size(1) + 1
            m = self.mask(L).to(memory.device)
            if t==0:
                cand = torch.cat([dec_in, memory[:, 0:1]], dim=1)
            else:
                tf = (random.random()<teacher_forcing_ratio)
                cand = torch.cat([dec_in, memory[:, t:t+1] if not tf else memory[:, t-1:t]], dim=1)
            out = self.dec(cand, memory, tgt_mask=m)
            preds.append(out[:, -1, :])
            dec_in = out
        return torch.stack(preds, dim=1)

class StateActionVQTransformer(nn.Module):
    def __init__(self, image_size=84, patch_size=4, action_dim=18, embed_dim=256, n_heads=8, enc_layers=4, dec_layers=4, ff=512, num_tokens=128, beta=1.0):
        super().__init__()
        self.patcher = PatchEmbedding(1, patch_size, embed_dim, image_size)
        self.aggregator = PatchAggregator(embed_dim)
        self.action_embed = nn.Embedding(action_dim, embed_dim)
        self.seq_enc = SeqEncoder(embed_dim, n_heads, enc_layers, ff)
        self.vq = VectorQuantizer(num_tokens, embed_dim, beta)
        self.seq_dec = SeqDecoder(embed_dim, n_heads, dec_layers, ff)
        self.frame_dec = FrameDecoder(embed_dim, patch_size, image_size)
        self.patch_size = patch_size
    def forward(self, states, actions, teacher_forcing_ratio=1.0):
        B, T, C, H, W = states.shape
        s2d = states.view(B*T, C, H, W)
        p = self.patcher(s2d)
        f_agg = self.aggregator(p).view(B, T, -1)
        a_1d = actions.view(-1)
        a_emb = self.action_embed(a_1d).view(B, T, -1)
        x = f_agg + a_emb
        enc_out = self.seq_enc(x)
        q, c_loss, cb_loss = self.vq(enc_out)
        dec_seq = self.seq_dec.forward_autoregressive(q, teacher_forcing_ratio, None)
        dec_seq_2d = dec_seq.view(B*T, -1)
        dec_seq_2d = dec_seq_2d.unsqueeze(1).expand(-1, p.size(1), -1)
        rec = self.frame_dec(dec_seq_2d).view(B, T, 1, H, W)
        r_loss = F.mse_loss(rec, states)
        loss = r_loss + c_loss + cb_loss
        return rec, loss, {"recon_loss":r_loss.item(),"commitment_loss":c_loss.item(),"codebook_loss":cb_loss.item()}

def train_model(model, dataset, epochs=10, bs=8, lr=1e-4, tf=1.0, device='cuda'):
    from torch.utils.data import DataLoader
    import torch.optim as optim
    dl = DataLoader(dataset, batch_size=bs, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for e in range(1, epochs+1):
        model.train()
        tot = 0
        for batch in dl:
            st = batch["states"].to(device)
            ac = batch["actions"].to(device)
            _, l, s = model(st, ac, tf)
            opt.zero_grad()
            l.backward()
            opt.step()
            tot += l.item()
        print(f"Epoch {e}/{epochs} Loss={tot/len(dl):.4f} Recon={s['recon_loss']:.4f} Commit={s['commitment_loss']:.4f} Codebook={s['codebook_loss']:.4f}")
        tf *= 0.95

def main():
    pass
