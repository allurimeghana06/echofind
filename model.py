"""Phase 2: encoder, projector, SSL losses (+ Extension A decoder).

AudioEncoder
  input : waveform (T,) / (B, T)          -> computes the log-mel itself
          or spectrogram (M, F) / (B, M, F) / (B, 1, M, F)
  output: h in R^512, L2-normalised by default (ready for cosine similarity)

  Any input length works (Extension C): the network is fully convolutional and ends in
  global average pooling, so 2 s and 60 s clips both give one 512-d vector, with no
  resizing or zero padding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import LogMel, N_MELS, instance_norm


# ----------------------------------------------------------------------------- ResNet-18
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.short is None else self.short(x)))


class AudioEncoder(nn.Module):
    """ResNet-18 for 1-channel log-mel spectrograms.

    Stem: 3x3 conv (stride 2) + 3x3 max-pool (stride 2) instead of ImageNet's 7x7.
    Total downsampling 32x: a 5 s clip (128 x 431) becomes a 512 x 4 x 14 feature map.
    """

    def __init__(self, dim=512):
        super().__init__()
        self.frontend = LogMel()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        chans = [64, 128, 256, dim]
        layers, cin = [], 64
        for i, c in enumerate(chans):
            layers.append(nn.Sequential(BasicBlock(cin, c, 1 if i == 0 else 2), BasicBlock(c, c)))
            cin = c
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.dim = dim
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    # -- input handling -------------------------------------------------------
    def to_spec(self, x):
        """Accept waveform or spectrogram, return normalised (B, 1, M, F)."""
        if x.dim() == 1:                                   # (T,) waveform
            return self.frontend(x)
        if x.dim() == 2:
            if x.shape[0] == N_MELS and x.shape[1] != N_MELS:   # (M, F) spectrogram
                return instance_norm(x[None, None])
            return self.frontend(x)                        # (B, T) waveforms
        if x.dim() == 3:                                   # (B, M, F) or (1, M, F)
            x = x.unsqueeze(1)
        return instance_norm(x.float())                    # idempotent if already normalised

    # -- forward variants -----------------------------------------------------
    def feature_maps(self, x):
        """All intermediate maps (used by the layer ablation and the decoder)."""
        s = self.to_spec(x)
        x0 = self.stem(s)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return {"layer1": x1, "layer2": x2, "layer3": x3, "layer4": x4}

    def embed(self, x, normalize=True, layer="layer4"):
        """Global-average-pooled embedding from a given layer."""
        if layer == "layer4":
            s = self.to_spec(x)
            f = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(s)))))
        else:
            f = self.feature_maps(x)[layer]
        h = f.mean(dim=(2, 3))
        return F.normalize(h, dim=-1) if normalize else h

    def forward(self, x):
        """h, L2-normalised. Training calls embed(x, normalize=False) instead."""
        return self.embed(x, normalize=True)


# ----------------------------------------------------------------------------- SSL heads
class Projector(nn.Module):
    """h -> z, used only by the loss and thrown away after training.

    Barlow Twins works better with a wide projector (the paper uses 8192); 2048 is a
    good balance for a T4. Deliberate change from the 512->512->128 first plan.
    """

    def __init__(self, din=512, hidden=2048, dout=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, dout, bias=False))

    def forward(self, h):
        return self.net(h)


def barlow_twins_loss(z1, z2, lam=5e-3):
    """Make the cross-correlation matrix of the two views' z close to the identity.

    diagonal -> 1   : each feature agrees across the two views (invariance)
    off-diag -> 0   : different features carry different information (no collapse)
    """
    z1, z2 = z1.float(), z2.float()
    N = z1.shape[0]
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-5)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-5)
    c = z1.T @ z2 / N
    on = (torch.diagonal(c) - 1).pow(2).sum()
    off = c.pow(2).sum() - torch.diagonal(c).pow(2).sum()
    return on + lam * off, on.detach(), off.detach()


def nt_xent_loss(z1, z2, tau=0.2):
    """SimCLR loss (backup objective). Positives: (i, i+N). Everything else: negatives."""
    z = F.normalize(torch.cat([z1, z2]).float(), dim=-1)
    N = z1.shape[0]
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, target)


# ----------------------------------------------------------------------------- diagnostics
@torch.no_grad()
def alignment_uniformity(h1, h2, t=2.0):
    """Wang & Isola (2020). Lower is better for both.
    alignment: how close positive pairs are; uniformity: how evenly spread all points are."""
    h1, h2 = F.normalize(h1.float(), dim=-1), F.normalize(h2.float(), dim=-1)
    align = (h1 - h2).norm(dim=1).pow(2).mean().item()
    sq = torch.pdist(h1).pow(2)
    unif = sq.mul(-t).exp().mean().log().item()
    return align, unif


@torch.no_grad()
def collapse_check(h):
    """Std of the normalised embedding per dimension. ~1/sqrt(512)=0.044 is healthy; ~0 = collapse."""
    return F.normalize(h.float(), dim=-1).std(0).mean().item()


# ----------------------------------------------------------------------------- Extension A
class SpecDecoder(nn.Module):
    """Decoder for Extension A: embedding -> normalised log-mel (B, 1, 128, frames).

    from_vector=True : input is the pooled 512-d vector h (what the spec literally asks).
                       All timing information is gone, so expect a static texture.
    from_vector=False: input is the 512 x 4 x T' map before pooling (keeps time). Used as
                       the comparison: does the pooled vector or the map carry the music?
    """

    def __init__(self, dim=512, n_mels=N_MELS, frames=431, from_vector=True):
        super().__init__()
        self.from_vector, self.frames, self.n_mels = from_vector, frames, n_mels
        self.t0 = (frames + 31) // 32
        if from_vector:
            self.fc = nn.Linear(dim, dim * 4 * self.t0)
        ch = [dim, 256, 128, 64, 32, 16]
        ups = []
        for a, b in zip(ch[:-1], ch[1:]):
            ups += [nn.ConvTranspose2d(a, b, 4, 2, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                    nn.Conv2d(b, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
        self.up = nn.Sequential(*ups)
        self.out = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, z):
        if self.from_vector:
            x = self.fc(z).view(z.shape[0], -1, 4, self.t0)
        else:
            x = z
        y = self.out(self.up(x))                         # (B, 1, 128, 32*T')
        return y[..., :self.n_mels, :self.frames]
