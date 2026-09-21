from pathlib import Path

root = Path(__file__).parent.parent                     # top-level directory;
DATA_PATH = root / "data/"                              # datasets and pretrained weights; 数据集和预训练权重文件;
TRAINING_PATH = root / "outputs/training/"              # training checkpoints; 训练权重;
EVAL_PATH = root / "outputs/results/"                   # evaluation results; 测试结果;
