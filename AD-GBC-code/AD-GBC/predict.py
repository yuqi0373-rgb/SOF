import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from adgbc import load_model
from adgbc.checkpoints import recommended_size
from adgbc.data import read_image, preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('prediction.png'))
    parser.add_argument('--size', type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threshold', type=float, default=.5)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error('threshold must be in (0,1)')
    size = args.size or recommended_size(args.weights)
    if size < 32 or size % 32:
        parser.error('size must be a positive multiple of 32')
    model = load_model(args.weights, args.device)
    image = read_image(args.image)
    x = preprocess(image, size)[None].to(args.device)
    with torch.inference_mode():
        probability = model(x)['logits'].sigmoid()[0, 0].cpu().numpy()
    if not np.isfinite(probability).all():
        raise RuntimeError('Nonfinite prediction')
    probability = cv2.resize(probability, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = (probability > args.threshold).astype(np.uint8) * 255
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), mask):
        raise RuntimeError(f'Cannot write mask: {args.output}')
    print(args.output)


if __name__ == '__main__':
    main()
