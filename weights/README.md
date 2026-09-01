# weights/

Checkpoints are not distributed with this repository. Train them with the steps in the top-level
`README.md` and place them here; `src/config.py` expects these names:

| file | what it is | produced by |
|---|---|---|
| `teacher_KAIST_10p.pth.tar` | supervised teacher, trained on the labeled subset only | `src/train_teacher.py` |
| `teacher_KAIST_10p_3way.pth.tar` | the same teacher with three prediction heads, if yours has a single head | `expand_teacher_to_3way.py` |
| `SSMPD_KAIST_10p.pth.tar` | the full model | `src/train.py` |

`.gitignore` keeps `*.pth.tar` out of the repository.

## Checking a checkpoint

Each file is a dict with `epoch`, `loss`, `model` and `optimizer`, where `model` is the module
object itself rather than a `state_dict`. Load it from inside `src/`:

```python
import torch
ck = torch.load('../weights/SSMPD_KAIST_10p.pth.tar', map_location='cpu')
net = ck['model']
print(type(net).__name__, list(net._modules.keys()))
print(sum(p.numel() for p in net.parameters()))
```

A full model is an `SSD300_3Way` with the submodules `base`, `pred_convs`, `pred_convs_vis` and
`pred_convs_lwir`, and 41,760 priors at a 512x640 input. A forward pass takes a 3-channel visible
tensor and a 1-channel thermal tensor and returns fusion, visible and thermal locations and scores
plus the fusion feature maps.
