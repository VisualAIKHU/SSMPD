# weights/

Checkpoints are not distributed with this repository. Train them with the steps in the top-level
`README.md` and place them here. `src/config.py` points at this directory:

| file | produced by |
|---|---|
| `teacher_KAIST_10p.pth.tar` | `src/train_teacher.py` |
| `SSMPD_KAIST_10p.pth.tar` | `src/train.py` |

Each file is a dict with `epoch`, `loss`, `model` and `optimizer`, where `model` is the module
object itself rather than a `state_dict`, so `torch.load` needs `src/model.py` importable. Run
training and inference from inside `src/`, or through `train.sh` and `Inference.sh`.

`.gitignore` keeps `*.pth.tar` out of the repository.
