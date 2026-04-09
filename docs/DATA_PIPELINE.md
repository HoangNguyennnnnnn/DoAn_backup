# Data Pipeline and Preprocessing Flow

## Overview

The data pipeline converts ModelNet40 3D mesh files (OFF format) into O-Voxel representations suitable for Shape SC-VAE training.

```
ModelNet40 Dataset (Kaggle)
    ↓
    └─ OFF Mesh Files (12,311 shapes, 40 categories)
    ↓
OFF File Loader
    ├─ Parse OFF format
    └─ Extract vertices and faces
    ↓
Mesh Normalization
    ├─ Center to origin
    └─ Scale to unit box [-0.5, 0.5]³
    ↓
Voxelization (OFF → O-Voxel)
    ├─ Rasterize mesh to 32³ voxel grid
    └─ Binary occupancy representation
    ↓
Cached O-Voxel Representation
    ├─ Store as .pt files
    └─ Load directly during training (skip mesh processing)
    ↓
PyTorch DataLoader
    ├─ Batch sampling
    ├─ Optional augmentation
    └─ GPU transfer
    ↓
Shape SC-VAE Training
```

---

## Stage 1: Data Source (ModelNet40)

### Dataset Structure

```
/kaggle/input/modelnet40-princeton-3d-object-dataset/
├── ModelNet40/
│   ├── airplane/
│   │   ├── train/
│   │   │   ├── airplane_0001.off
│   │   │   ├── airplane_0002.off
│   │   │   └── ...
│   │   ├── test/
│   │   │   ├── airplane_0001.off
│   │   │   └── ...
│   │   └── ...
│   ├── bathtub/
│   ├── bed/
│   ├── ... (40 categories total)
│   └── README.txt
├── split/
│   ├── modelnet40_train.txt
│   ├── modelnet40_test.txt
│   └── ...
└── [metadata files]
```

### Dataset Statistics

| Property         | Value                   |
| ---------------- | ----------------------- |
| Total shapes     | 12,311                  |
| Categories       | 40 object classes       |
| Train/Test split | ~80/20 (typical)        |
| Format           | OFF (Wavefront-like)    |
| Mesh quality     | High-quality CAD models |

---

## Stage 2: OFF File Format

### OFF Specification

OFF (Object File Format) is a simple text format for 3D geometry:

```
OFF
<num_vertices> <num_faces> <num_edges>
<x0> <y0> <z0>
<x1> <y1> <z1>
...
<xn> <yn> <zn>
<num_verts_in_face0> <v0> <v1> <v2> ...
<num_verts_in_face1> <v0> <v1> <v2> ...
...
<num_verts_in_facem> <v0> <v1> <v2> ...
```

### Example OFF File

```
OFF
4 2 0
0 0 0
1 0 0
1 1 0
0 1 0
3 0 1 2
3 0 2 3
```

(This defines a square with 4 vertices and 2 triangular faces)

### Loading in Python

```python
from src.data.voxel_converter import OFFMeshLoader

vertices, faces = OFFMeshLoader.load_off("airplane_0001.off")
# vertices: (N, 3) array
# faces: (M, 3) array of triangle indices
```

---

## Stage 3: Mesh Normalization

### Purpose

Normalize mesh to a standard coordinate system to ensure:

- Consistent voxelization across all meshes
- Proper alignment for rotation/scaling invariance
- Reproducible preprocessing

### Normalization Steps

1. **Center at Origin**

   ```python
   centroid = vertices.mean(axis=0)
   vertices_centered = vertices - centroid
   ```

2. **Scale to Unit Box**

   ```python
   # Find bounding box
   bbox_min = vertices.min(axis=0)
   bbox_max = vertices.max(axis=0)
   bbox_size = bbox_max - bbox_min

   # Scale to fit in [-0.5, 0.5]³
   max_extent = bbox_size.max()
   vertices_normalized = vertices_centered / max_extent
   ```

3. **Final Result**
   ```python
   # All vertices now in [-0.5, 0.5]³
   assert vertices_normalized.min() >= -0.5
   assert vertices_normalized.max() <= 0.5
   ```

### Normalization Code

```python
from src.data.voxel_converter import MeshNormalizer

vertices_norm = MeshNormalizer.normalize(vertices)
# Output: vertices in [-0.5, 0.5]³
```

---

## Stage 4: Voxelization (OFF → O-Voxel)

### Voxel Grid Representation

O-Voxel (Occupancy Voxel) is a 3D binary grid:

```
Shape: (1, 32, 32, 32)
├─ 1 channel: Binary occupancy (0 = empty, 1 = occupied)
├─ 32³ = 32,768 voxels
└─ dtype: float32 (for torch.nn operations)
```

### Voxelization Methods

#### Method 1: Ray Casting (Scanline)

- Ray cast from one side through mesh
- Count triangle intersections
- Voxel occupied if odd number of intersections

**Pros**: Numerically stable, handles complex geometry
**Cons**: Computationally expensive

#### Method 2: Signed Distance Function (SDF)

- Compute signed distance from each voxel center to mesh
- Threshold at 0: negative = inside, positive = outside

**Pros**: Smooth gradients, smooth surface
**Cons**: Requires more computation

#### Method 3: Triangle Rasterization

- Rasterize each triangle to voxel grid
- Mark voxels inside triangles as occupied

**Pros**: Fast, direct
**Cons**: May miss thin structures

### Recommended: Trimesh Library

For production quality, use trimesh library:

```python
import trimesh
from src.data.voxel_converter import TrimeshVoxelizer

# Load mesh
mesh = trimesh.load("airplane_0001.off")

# Voxelize
voxel_grid = TrimeshVoxelizer.voxelize_with_trimesh(
    mesh_file="airplane_0001.off",
    resolution=32,
)
```

Trimesh provides:

- Robust OFF parsing
- High-quality voxelization
- Signed distance support
- Built-in mesh validation

### Implementation

```python
from src.data.voxel_converter import VoxelConverter

converter = VoxelConverter(resolution=32)
voxel_grid = converter.voxelize(vertices_norm, faces)
# Output: (1, 32, 32, 32) tensor
```

---

## Stage 5: Caching Strategy

### Cache Directory

```
/kaggle/working/cache/
├── stage1_voxels/
│   ├── airplane/
│   │   ├── airplane_0001.pt
│   │   ├── airplane_0002.pt
│   │   └── ...
│   ├── bathtub/
│   ├── ...
│   └── metadata.json  # {category: num_samples}
└── preprocessing_log.txt  # Which meshes were successfully voxelized
```

### Cache Files

**Voxel file** (`airplane_0001.pt`):

```python
# Saved with torch.save()
cache_data = {
    "voxel": tensor(shape=[1, 32, 32, 32]),
    "category": "airplane",
    "category_idx": 0,
    "source_file": "airplane_0001.off",
}
torch.save(cache_data, "cache/stage1_voxels/airplane/airplane_0001.pt")
```

### Caching Benefits

- **Speed**: 1st epoch takes ~2x longer (voxelization); 2+ epochs use cache
- **Reproducibility**: Same voxels used across training runs
- **Debugging**: Can inspect voxel quality offline

### Cache Loading

```python
from src.data.modelnet40_loader import ModelNet40Dataset

dataset = ModelNet40Dataset(
    dataset_root="/kaggle/input/modelnet40-...",
    cache_voxels=True,  # Enable caching
)

# First epoch: Slow (voxelizes meshes)
# Subsequent epochs: Fast (loads .pt files)
```

---

## Stage 6: PyTorch DataLoader

### Dataset Class

```python
from torch.utils.data import Dataset, DataLoader

class ModelNet40Dataset(Dataset):
    def __init__(self, dataset_root, split="train", cache_voxels=True):
        # Load file list and split assignments
        # Setup cache directory
        pass

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load voxel (from cache or voxelize on-the-fly)
        # Return (voxel, category_idx)
        return voxel, category_idx
```

### DataLoader Configuration

**Training**:

```python
from src.data.modelnet40_loader import DataLoaderFactory

train_loader = DataLoaderFactory.create_train_loader(
    dataset_root="/kaggle/input/modelnet40-...",
    batch_size=8,           # From hardware config
    num_workers=4,          # From hardware config
    shuffle=True,           # Training-specific
    pin_memory=True,        # GPU transfer optimization
)
```

**Validation**:

```python
val_loader = DataLoaderFactory.create_val_loader(
    dataset_root="/kaggle/input/modelnet40-...",
    batch_size=16,          # Can be larger
    num_workers=2,          # Fewer workers
    shuffle=False,          # No shuffling needed
)
```

### Batch Structure

```python
for batch_idx, (voxels, categories) in enumerate(train_loader):
    # voxels: (B, 1, 32, 32, 32) batch of voxel grids
    # categories: (B,) batch of category indices

    # Transfer to GPU
    voxels = voxels.to(device)
    categories = categories.to(device)

    # Forward pass through Shape SC-VAE
    mu, log_var = encoder(voxels)
    z = sampler(mu, log_var)
    recon = decoder(z)

    # Compute loss
    loss = reconstruction_loss(recon, voxels) + kl_loss(mu, log_var)
```

---

## Stage 7: Optional Augmentation

### Augmentation Strategies (Stage 1)

Currently **disabled** for clean shape pretraining:

```yaml
augmentation:
  enabled: false # Clean shapes for baseline
```

### Future Augmentation (Stage 2+)

Enable optional augmentation:

```yaml
augmentation:
  enabled: true
  random_rotation: true
  rotation_range: [0, 360]
  random_scale: true
  scale_range: [0.8, 1.2]
  random_jitter: false
```

### Augmentation Implementation

```python
def augment_voxel(voxel, rotation_range, scale_range):
    # 1. Random rotation around z-axis
    angle = random.uniform(*rotation_range)
    voxel = rotate_voxel(voxel, angle)

    # 2. Random scaling
    scale = random.uniform(*scale_range)
    # Interpolate to new resolution and pad/crop
    voxel = scale_voxel(voxel, scale)

    return voxel
```

---

## Troubleshooting

### Problem: "Dataset not found"

```
FileNotFoundError: /kaggle/input/modelnet40-princeton-...
```

**Solution**:

1. Attach dataset in Kaggle notebook UI (`Add data`)
2. Verify exact slug in Kaggle dataset page
3. Check `DATASET_ROOT` environment variable

### Problem: "Voxelization takes too long"

**Solution**:

1. Enable caching: `cache_voxels=True`
2. First epoch will be slower; subsequent epochs use cache
3. Monitor cache directory size: `du -sh /kaggle/working/cache/`

### Problem: "Out of memory during voxelization"

**Solution**:

1. Process fewer meshes per epoch: Reduce `batch_size`
2. Use lower resolution: `target_resolution=16` (16³ instead of 32³)
3. Voxelize offline and save cache first

### Problem: "Mesh file corrupted"

**Solution**:

1. Check OFF file format with `head -20 file.off`
2. Validate using trimesh: `mesh = trimesh.load(file)`
3. Skip corrupted files and continue training

---

## Performance Metrics

### Data Loading Benchmarks

| Operation       | Time       | Hardware          |
| --------------- | ---------- | ----------------- |
| Load 1 OFF file | ~10 ms     | CPU               |
| Normalize mesh  | ~5 ms      | CPU               |
| Voxelize (32³)  | ~50 ms     | CPU               |
| Total 1st epoch | ~10-15 min | P100 (first pass) |
| Total 2+ epochs | ~2-3 min   | P100 (cached)     |

### Memory Usage

| Component               | Memory  | Notes                |
| ----------------------- | ------- | -------------------- |
| Voxel tensor (1 sample) | 128 KB  | 32³ grid             |
| Batch (8 samples)       | 1 MB    | 8× voxel size        |
| Cache (12,311 samples)  | ~1.5 GB | All voxels cached    |
| Worker buffers          | ~1-2 GB | 4 workers × prefetch |

---

## See Also

- [README.md](../README.md) — Project overview
- [KAGGLE_RUNBOOK.md](KAGGLE_RUNBOOK.md) — Kaggle execution
- [src/data/](../src/data/) — Data loading implementation
- [configs/data_stage1.yaml](../configs/data_stage1.yaml) — Data configuration
