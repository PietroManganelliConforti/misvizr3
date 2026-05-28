import torch, json
import numpy as np

train = torch.load('data/precomp/misviz_synth/train_small_misviz_synth_tinychart_embedded_images.pt', map_location='cpu')

# Load metadata to get labels
data = json.load(open('data/misviz_synth/misviz_synth.json'))
train_meta = [d for d in data if d['split'] == 'train_small']
labels = [d['misleader'][0] if d['misleader'] else 'no misleader' for d in train_meta]

# Cosine sim WITHIN same class vs BETWEEN different classes
from collections import defaultdict
by_class = defaultdict(list)
for i, l in enumerate(labels):
    by_class[l].append(i)

within_sims = []
between_sims = []
np.random.seed(42)

for cls, idxs in by_class.items():
    if len(idxs) < 2:
        continue
    # Within-class: 50 random pairs
    for _ in range(min(50, len(idxs))):
        i, j = np.random.choice(idxs, 2, replace=False)
        sim = torch.nn.functional.cosine_similarity(train[i:i+1], train[j:j+1]).item()
        within_sims.append(sim)

# Between-class: 200 random pairs from different classes
all_labels_arr = np.array(labels)
for _ in range(200):
    i, j = np.random.choice(len(labels), 2, replace=False)
    if labels[i] != labels[j]:
        sim = torch.nn.functional.cosine_similarity(train[i:i+1], train[j:j+1]).item()
        between_sims.append(sim)

print(f'Within-class  cosine sim: mean={np.mean(within_sims):.4f} std={np.std(within_sims):.4f}')
print(f'Between-class cosine sim: mean={np.mean(between_sims):.4f} std={np.std(between_sims):.4f}')
print(f'Difference: {np.mean(within_sims) - np.mean(between_sims):.4f}')
print()

# Per-class stats
for cls in sorted(by_class.keys()):
    idxs = by_class[cls]
    norms = train[idxs].norm(dim=1)
    print(f'{cls:<40} n={len(idxs):>4}  norm={norms.mean():.2f}±{norms.std():.2f}')
"