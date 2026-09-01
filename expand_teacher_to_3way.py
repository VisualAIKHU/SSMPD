"""Turn a supervised SSD300 teacher into the SSD300_3Way model SSMPD trains with.

The supervised teacher has a single fusion prediction head, while SSMPD needs a
model with three heads (fusion, visible, thermal), because UMAS supervises all
three modalities. This script copies the shared backbone and duplicates the
fusion head into the visible and thermal heads, which is how the first
semi-supervised stage is started.

Skip this if you trained the teacher with `src/train_teacher.py`: that script
already builds SSD300_3Way.

Usage:
    python expand_teacher_to_3way.py \
        --input  weights/teacher_KAIST_10p.pth.tar \
        --output weights/teacher_KAIST_10p_3way.pth.tar
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='supervised SSD300 checkpoint')
    parser.add_argument('--output', required=True, help='where to write the SSD300_3Way checkpoint')
    parser.add_argument('--n-classes', type=int, default=3)
    args = parser.parse_args()

    from model import SSD300_3Way

    ckpt = torch.load(args.input, map_location='cpu')
    src = ckpt['model']
    src_sd = src.state_dict()

    dst = SSD300_3Way(n_classes=args.n_classes)
    dst_sd = dst.state_dict()

    copied, duplicated, skipped = 0, 0, []
    new_sd = {}
    for k, v in dst_sd.items():
        if k in src_sd and src_sd[k].shape == v.shape:
            new_sd[k] = src_sd[k].clone()
            copied += 1
        elif k.startswith(('pred_convs_vis.', 'pred_convs_lwir.')):
            # duplicate the single fusion head into the two modality heads
            base_key = 'pred_convs.' + k.split('.', 1)[1]
            if base_key in src_sd and src_sd[base_key].shape == v.shape:
                new_sd[k] = src_sd[base_key].clone()
                duplicated += 1
            else:
                new_sd[k] = v
                skipped.append(k)
        else:
            new_sd[k] = v
            skipped.append(k)

    dst.load_state_dict(new_sd)

    torch.save({'epoch': ckpt.get('epoch', 0),
                'loss': ckpt.get('loss', 0.0),
                'model': dst,
                'optimizer': None},
               args.output)

    print(f'copied from the supervised teacher: {copied} tensors')
    print(f'duplicated into the visible/thermal heads: {duplicated} tensors')
    print(f'left at initialization: {len(skipped)} tensors')
    for k in skipped[:10]:
        print(f'  {k}')
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
