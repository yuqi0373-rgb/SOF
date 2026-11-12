# AD-GBC

Binary medical image segmentation with Rolling-UNet and U-KAN backbones.

## Installation

Use Python 3.10 or newer and install a matching PyTorch/torchvision build for your platform.

Download `AD-GBC-weights.zip` from this repository's Releases and place its `weights` directory beside `predict.py`. The supplied code and weight archives both use an `AD-GBC/` root directory, so extracting both archives into the same parent directory merges them into a complete project.

For a Git LFS checkout, `git lfs pull` can also download checkpoints tracked by the repository. The release archive does not require Git LFS.

```bash
pip install -r requirements.txt
```

## Inference

```bash
python predict.py \
  --weights weights/rolling/busi/split_41.ckpt \
  --image /path/to/image.png \
  --output prediction.png \
  --device cpu
```

The backbone is detected from the weight tensors. The input resolution is read from `weights/index.json`: 512 for GlaS and 256 for the other datasets. Use `--size` when loading a checkpoint outside the weights directory. Images use OpenCV BGR channel order and ImageNet mean/std normalization, matching the supplied models. Masks are resized to the original image dimensions and thresholded at 0.5.

```python
import torch
from adgbc import load_model
from adgbc.data import read_image, preprocess

model = load_model('weights/ukan/cvc/split_41.ckpt', device='cpu')
image = preprocess(read_image('/path/to/image.png'), 256)[None]
with torch.inference_mode():
    probability = model(image)['logits'].sigmoid()
```

Weights are grouped by backbone and dataset. `split_41`, `split_2981`, and `split_6142` identify separate dataset partitions. Additional `reference_*.ckpt` files provide the retained BUSI/GlaS model variants. All files are tensor state dictionaries; strict loading is used.

## Training

Arrange binary masks under `masks/0`:

```text
inputs/
  busi/
    images/example.png
    masks/0/example_mask.png
  glas/
    images/example.png
    masks/0/example.png
  cvc/
    images/example.png
    masks/0/example.png
  ISIC17/
    images/example.jpg
    masks/0/example_segmentation.png
```

```bash
python train.py --dataset busi --backbone rolling --data-root inputs
python train.py --dataset glas --backbone ukan --data-root inputs --batch-size 2
```

The training entry point starts from random initialization and runs the requested number of epochs. Defaults are 400 epochs, Adam with learning rate 1e-4 and weight decay 1e-4, cosine decay to 1e-5, BCE+Dice loss, and an 80/20 split with seed 41. The best validation Dice model and the final model are saved as `runs/<backbone>/<dataset>/model.ckpt` and `last.ckpt`.

The supplied checkpoints are for inference; the training entry point provides the default fixed-length training workflow.

## Acknowledgements

Backbone implementations are derived from [Rolling-UNet](https://github.com/Jiaoyang45/Rolling-Unet) and [U-KAN](https://github.com/CUHK-AIM-Group/U-KAN). See `NOTICE` and `LICENSE`.
