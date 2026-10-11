"""Phase 2: two-head encoder, SSL losses, nearest-neighbour queue, diagnostics
(+ Extension A decoder).

Architecture (one shared trunk, two label-free heads):

  waveform -> log-mel -> ResNet-18 trunk -> GAP -> h (512, shared)
                                                   |-> identity head -> e_id (d_id)
                                                   |-> style head    -> e_st (d_st)
  full embedding e = [ e_id/|e_id| ; e_st/|e_st| ] / sqrt(2)      (unit length)

  d_id + d_st = 512 for the split models; single-head baselines use one head of 512.
  Projectors (loss only, discarded after training) sit on top of each head.

AudioEncoder input: waveform (T,)/(B, T) or spectrogram (M, F)/(B, M, F)/(B, 1, M, F).
Any input length works: the trunk is fully convolutional and ends in global average
pooling (Extension C).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import LogMel, N_MELS, instance_norm


# ----------------------------------------------------------------------------- trunk
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


class MLPHead(nn.Module):
    """Head that produces a subspace: 512 -> 512 -> d."""

    def __init__(self, din, dout, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden, bias=False), nn.BatchNorm1d(hidden),
                                 nn.ReLU(inplace=True), nn.Linear(hidden, dout))

    def forward(self, x):
        bn = self.net[1]
        if self.training and x.shape[0] == 1:          # a single input (e.g. the organisers'
            bn.eval()                                  # random-tensor check on a fresh model):
            try:                                       # use running statistics instead of
                return self.net(x)                     # batch statistics, which need >1 sample
            finally:
                bn.train()
        return self.net(x)


class AudioEncoder(nn.Module):
    """ResNet-18 trunk for 1-channel log-mels + identity head and/or style head.

    d_id, d_st: subspace sizes. (256, 256) = S-50; (512, 0) = M-id; (0, 512) = M-st.
    Stem: 3x3 conv (stride 2) + max-pool (stride 2); total downsampling 32x, so a 5 s clip
    (128 x 431) becomes a 512 x 4 x 14 map before pooling.
    """

    def __init__(self, d_id=256, d_st=256, trunk_dim=512):
        super().__init__()
        assert d_id > 0 or d_st > 0
        self.frontend = LogMel()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))
        chans = [64, 128, 256, trunk_dim]
        layers, cin = [], 64
        for i, c in enumerate(chans):
            layers.append(nn.Sequential(BasicBlock(cin, c, 1 if i == 0 else 2), BasicBlock(c, c)))
            cin = c
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.head_id = MLPHead(trunk_dim, d_id) if d_id > 0 else None
        self.head_st = MLPHead(trunk_dim, d_st) if d_st > 0 else None
        self.d_id, self.d_st, self.trunk_dim = d_id, d_st, trunk_dim
        self.dim = d_id + d_st
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    @property
    def spaces(self):
        """Embedding spaces this model has: 'id', 'st', 'full' (and 'trunk' for ablations)."""
        s = (["id"] if self.head_id is not None else []) + (["st"] if self.head_st is not None else [])
        return s + ["full"]

    def config(self):
        return {"d_id": self.d_id, "d_st": self.d_st, "trunk_dim": self.trunk_dim}

    # -- input handling -------------------------------------------------------
    def to_spec(self, x):
        if x.dim() == 1:                                   # (T,) waveform
            return self.frontend(x)
        if x.dim() == 2:
            if x.shape[0] == N_MELS and x.shape[1] != N_MELS:   # (M, F) spectrogram
                return instance_norm(x[None, None].float())
            return self.frontend(x)                        # (B, T) waveforms
        if x.dim() == 3:                                   # (B, M, F)
            x = x.unsqueeze(1)
        return instance_norm(x.float())

    # -- forward variants -----------------------------------------------------
    def feature_maps(self, x):
        s = self.to_spec(x)
        x1 = self.layer1(self.stem(s))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return {"layer1": x1, "layer2": x2, "layer3": x3, "layer4": x4}

    def trunk(self, x):
        """Pooled shared representation h (B, 512), not normalised."""
        s = self.to_spec(x)
        f = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(s)))))
        return f.mean(dim=(2, 3))

    def subspaces(self, x):
        """Raw (unnormalised) head outputs: {'h', 'id', 'st'} (missing heads -> None)."""
        h = self.trunk(x)
        return {"h": h,
                "id": self.head_id(h) if self.head_id is not None else None,
                "st": self.head_st(h) if self.head_st is not None else None}

    @staticmethod
    def combine(sub, space="full"):
        """Turn raw subspaces into the requested L2-normalised embedding."""
        if space == "trunk":
            return F.normalize(sub["h"], dim=-1)
        if space in ("id", "st"):
            if sub[space] is None:
                raise ValueError(f"this model has no '{space}' head")
            return F.normalize(sub[space], dim=-1)
        parts = [F.normalize(sub[k], dim=-1) for k in ("id", "st") if sub[k] is not None]
        return torch.cat(parts, dim=-1) / math.sqrt(len(parts))

    def embed(self, x, space="full"):
        return self.combine(self.subspaces(x), space)

    def forward(self, x):
        """Full embedding e = [e_id ; e_st], unit length (what get_embedding returns)."""
        return self.embed(x, "full")


# ----------------------------------------------------------------------------- SSL heads
class Projector(nn.Module):
    """Subspace -> z for the contrastive loss; thrown away after training."""

    def __init__(self, din, hidden=1024, dout=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden, bias=False), nn.BatchNorm1d(hidden),
                                 nn.ReLU(inplace=True), nn.Linear(hidden, dout))

    def forward(self, x):
        return self.net(x)


def nt_xent(z1, z2, tau):
    """SimCLR loss over 2N views: positives (i, i+N), all other views negatives."""
    z = F.normalize(torch.cat([z1, z2]).float(), dim=-1)
    N = z1.shape[0]
    sim = z @ z.T / tau
    sim.fill_diagonal_(float("-inf"))
    target = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, target)


def info_nce(q, k, tau):
    """One-directional InfoNCE: q_i should match k_i among all k in the batch."""
    q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
    logits = q @ k.T / tau
    return F.cross_entropy(logits, torch.arange(len(q), device=q.device))


class NNQueue:
    """Support set for nearest-neighbour positives (NNCLR, Dwibedi et al. 2021).

    Stores past style projections with their track ids. nearest() returns, for each
    query, the most similar stored embedding from a DIFFERENT track: the pull towards
    other, similar-sounding songs is what turns track-level discrimination into style
    clustering. Labels are never used.
    """

    def __init__(self, size, dim, device):
        self.z = torch.zeros(size, dim, device=device)
        self.ids = torch.full((size,), -1, dtype=torch.long, device=device)
        self.ptr, self.size, self.filled = 0, size, 0

    @torch.no_grad()
    def add(self, z, ids):
        z = F.normalize(z.float(), dim=-1)
        n = len(z)
        idx = (torch.arange(n, device=z.device) + self.ptr) % self.size
        self.z[idx], self.ids[idx] = z, ids.to(self.ids.device)
        self.ptr = int((self.ptr + n) % self.size)
        self.filled = min(self.size, self.filled + n)

    @torch.no_grad()
    def nearest(self, z, ids):
        zq = F.normalize(z.float(), dim=-1)
        sim = zq @ self.z.T
        if self.filled < self.size:                          # empty slots are not candidates
            sim[:, self.filled:] = float("-inf")
        sim[ids.view(-1, 1) == self.ids.view(1, -1)] = float("-inf")    # never the same track
        return self.z[sim.argmax(dim=1)]

    def ready(self, n):
        return self.filled >= n


def style_loss(s1, s3, tau, queue=None, ids=None, p_nn=0.0):
    """Symmetric style InfoNCE between v1 and v3 (distant segment of the same track).
    With probability p_nn per sample, the anchor is replaced by its nearest neighbour
    from another track in the support queue (NNCLR)."""
    if queue is not None and p_nn > 0 and queue.ready(len(s1)):
        m = (torch.rand(len(s1), 1, device=s1.device) < p_nn)
        q1 = torch.where(m, queue.nearest(s1, ids).to(s1.dtype), s1)
        q3 = torch.where(m, queue.nearest(s3, ids).to(s3.dtype), s3)
    else:
        q1, q3 = s1, s3
    return 0.5 * (info_nce(q1, s3, tau) + info_nce(q3, s1, tau))


def orthogonality_penalty(a, b):
    """Linear CKA between the identity and style subspaces of a batch (features
    standardised first). 1 = the two subspaces carry the same linear information (even if
    rotated or duplicated), 0 = linearly unrelated. a: (N, d_id), b: (N, d_st).

    Bounded in [0, 1] and not lowered by collapsing features. With N samples and
    d-dimensional subspaces, independent features still score ~d/(N+d) from sampling
    noise (about 0.45 at N=288, d=256), so per-batch values are only comparable at equal
    batch size and dims; the floor is a constant that does not bias the gradient.
    Report validation-set CKA (thousands of windows) as the measure of shared information."""
    a, b = a.float(), b.float()
    n = a.shape[0]
    a = (a - a.mean(0)) / (a.std(0) + 1e-5)
    b = (b - b.mean(0)) / (b.std(0) + 1e-5)
    hab = (a.T @ b / n).pow(2).sum()
    haa = (a.T @ a / n).pow(2).sum()
    hbb = (b.T @ b / n).pow(2).sum()
    return hab / (haa * hbb).sqrt()


def nn_probability(epoch, start=10, ramp=10, p_max=0.5):
    """Share of style anchors replaced by nearest neighbours: 0 before `start`, then a
    linear ramp to p_max over `ramp` epochs."""
    if epoch < start:
        return 0.0
    return p_max * min(1.0, (epoch - start + 1) / ramp)


# ----------------------------------------------------------------------------- diagnostics
@torch.no_grad()
def alignment_uniformity(h1, h2, t=2.0):
    """Wang & Isola (2020). Lower is better for both."""
    h1, h2 = F.normalize(h1.float(), dim=-1), F.normalize(h2.float(), dim=-1)
    align = (h1 - h2).norm(dim=1).pow(2).mean().item()
    unif = torch.pdist(h1).pow(2).mul(-t).exp().mean().log().item()
    return align, unif


@torch.no_grad()
def collapse_check(h):
    """Mean per-dimension std of the normalised embedding; ~1/sqrt(d) healthy, ~0 collapse."""
    return F.normalize(h.float(), dim=-1).std(0).mean().item()


def linear_cka(X, Y):
    """Linear CKA (Kornblith et al. 2019) between two sets of embeddings of the same
    inputs. 1 = same information (up to rotation/scale), 0 = unrelated."""
    X = torch.as_tensor(X, dtype=torch.float64)
    Y = torch.as_tensor(Y, dtype=torch.float64)
    X, Y = X - X.mean(0), Y - Y.mean(0)
    hsic = (X.T @ Y).pow(2).sum()
    return float(hsic / ((X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt()))


# ----------------------------------------------------------------------------- Extension A
class SpecDecoder(nn.Module):
    """Decoder for Extension A: embedding -> normalised log-mel (B, 1, 128, frames)."""

    def __init__(self, dim=512, n_mels=N_MELS, frames=431, from_vector=True):
        super().__init__()
        self.from_vector, self.frames, self.n_mels = from_vector, frames, n_mels
        self.t0 = (frames + 31) // 32
        if from_vector:
            self.fc = nn.Linear(dim, 512 * 4 * self.t0)
        ch = [512, 256, 128, 64, 32, 16]
        ups = []
        for a, b in zip(ch[:-1], ch[1:]):
            ups += [nn.ConvTranspose2d(a, b, 4, 2, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                    nn.Conv2d(b, b, 3, 1, 1), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
        self.up = nn.Sequential(*ups)
        self.out = nn.Conv2d(16, 1, 3, 1, 1)

    def forward(self, z):
        x = self.fc(z).view(z.shape[0], 512, 4, self.t0) if self.from_vector else z
        return self.out(self.up(x))[..., :self.n_mels, :self.frames]
