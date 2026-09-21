# AerialFusion

This repository contains a cleaned open-source package for AerialFusion registration and infrared-visible image fusion. The release keeps the model code, required inference modules, configuration, and pretrained weights needed by the provided demo. Local experiment outputs, logs, cache files, and dataset-specific test scripts are removed.

## Directory layout

```text
configs/                  model configuration
models/extractors/         SuperPoint extractor
models/matchers/AerialFusion/
                           AerialFusion matcher and dynamic fusion network
demo.py                    folder-based inference script
requirements.txt           Python dependencies
```

## Quick start

```bash
cd AerialFusion_open
pip install -r requirements.txt
python demo.py \
  --input_1 /path/to/visible_images \
  --input_2 /path/to/infrared_images \
  --output_dir outputs/demo \
  --resize 640 480
```

Use `CUDA_VISIBLE_DEVICES` to select a GPU, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python demo.py --input_1 /path/to/vi --input_2 /path/to/ir
```

## Weights

The demo expects these files in the repository tree:

```text
models/extractors/weights/superpoint_v1.pth
models/matchers/AerialFusion/weights/checkpoint_best.tar
models/matchers/AerialFusion/Dynamic_Process/Dynamic_Fusion_Net/checkpoint/epoch_72/model-72.pt
```

The copied local package currently keeps these files so that inference can run directly. If you publish to a platform with file-size or license restrictions, remove the binary weights and provide download instructions instead.

## Notes

- The original local experiment outputs are not included.
- Dataset paths are not hard-coded; pass input folders through command-line arguments.
- The demo saves fused color images to `--output_dir`.

## If you find this work helpful, please cite it as follows:

```bibtex
@inproceedings{qiu2026aerialfusion,
  title={AerialFusion: Co-Motion-Driven Unified Registration and Fusion on Multi-modal Data Streams from Aerial View},
  author={Qiu, Junhui and Xiang, Xiang and Wang, Hongyun and Gui, Jiaqi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={10},
  pages={8583--8591},
  year={2026}
}
```
