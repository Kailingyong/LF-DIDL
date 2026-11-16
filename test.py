"""Evaluate the pretrained 5x5, 4x LF-DIDL main model on HDF5 test sets."""
import argparse
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from model.LF_DIDL import LF_DIDL
from utils.dataloader_newdata import MultiTestSetDataLoader
from utils.utility import LFdivide, LFintegrate, cal_metrics


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--testset_dir', type=Path, required=True,
                        help='parent directory containing test set subdirectories of HDF5 files')
    parser.add_argument('--model_path', type=Path,
                        default=ROOT / 'weights/LF-DIDL_4xSR_epoch_100.tar')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--angRes', type=int, default=5)
    parser.add_argument('--upfactor', type=int, default=4)
    parser.add_argument('--patchsize_test', type=int, default=32)
    parser.add_argument('--minibatch_test', type=int, default=10)
    return parser.parse_args()


@torch.no_grad()
def evaluate_scene(net, data, label, args):
    # DataLoader batch size is one. LFdivide expects [U, V, C, H, W].
    data, label = data.squeeze(0), label.squeeze(0)
    size = args.patchsize_test
    patches = LFdivide(data, size, size // 2)
    n1, n2 = patches.shape[:2]
    patches = rearrange(patches, 'n1 n2 u v c h w -> (n1 n2) u v c h w')

    outputs = []
    for start in range(0, patches.shape[0], args.minibatch_test):
        batch = patches[start:start + args.minibatch_test].to(args.device)
        # In evaluation mode the model generates its condition from LR input;
        # its second tuple element is unused.
        outputs.append(net((batch, None, True)).cpu())
    outputs = rearrange(torch.cat(outputs), '(n1 n2) u v c h w -> n1 n2 u v c h w',
                        n1=n1, n2=n2)
    output = LFintegrate(outputs, size * args.upfactor,
                         size * args.upfactor // 2)
    output = output[..., :data.shape[-2] * args.upfactor,
                    :data.shape[-1] * args.upfactor]
    return cal_metrics(label, output)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('LF-DIDL requires CUDA; no CUDA device is available.')
    if not args.testset_dir.is_dir():
        raise FileNotFoundError(args.testset_dir)
    if not args.model_path.is_file():
        raise FileNotFoundError(args.model_path)
    if args.minibatch_test < 1:
        raise ValueError('--minibatch_test must be positive')

    net = LF_DIDL(args).to(args.device)
    checkpoint = torch.load(args.model_path, map_location=args.device,
                            weights_only=True)
    net.load_state_dict(checkpoint['state_dict'], strict=True)
    net.eval()

    names, loaders, _ = MultiTestSetDataLoader(args)
    scores = []
    for name, loader in zip(names, loaders):
        scene_scores = [evaluate_scene(net, data, label, args)
                        for data, label in tqdm(loader, desc=name)]
        psnr, ssim = np.mean(scene_scores, axis=0)
        print(f'{name}: PSNR {psnr:.3f}, SSIM {ssim:.4f}')
        scores.append((psnr, ssim))
    if scores:
        psnr, ssim = np.mean(scores, axis=0)
        print(f'Mean over datasets: PSNR {psnr:.3f}, SSIM {ssim:.4f}')


if __name__ == '__main__':
    main()
