# EchoFind — self-supervised audio retrieval (IEEE Impulse 2026)

Barlow Twins on FMA-small. A ResNet-18 encoder maps any-length audio to a 512-d unit vector.
Two views of a track = two 5 s crops from different moments, each passed through a simulated
noisy room (reverb, traffic + babble noise, EQ, gain, SpecAugment).

| Spec item | Where |
| --- | --- |
| Log-mel (differentiable) | `features.py` |
| `AugmentationPipeline` | `augment.py` |
| SSL Dataset | `data.py` |
| `AudioEncoder`, loss | `model.py` |
| `get_embedding`, `predict_track` | `submission.py` |
| Training logs, t-SNE, experiments | `notebooks/` |
| Weights, test-set index | `weights/` |

```python
from submission import get_embedding, predict_track
predict_track("mystery_query.wav")            # uses weights/database.npz (FMA-small test set)
predict_track("q.wav", database={...})        # or your own {track_id: vector} dict / .npz / folder
```
