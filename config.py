from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data.nosync"
DATASET_RAW_PATH = DATA_DIR / "raw"
DATASET_PATH = DATA_DIR / "dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints.nosync"
LOG_DIR = PROJECT_ROOT / "logs.nosync"
NORM_STATS_PATH = DATA_DIR / "norm_stats.json"

HF_DATASET = "OpenMed/pgc-schizophrenia"
HF_CONFIG = "scz2022"

NUMERIC_FEATURES = [
    "BP_norm", "FRQ_A", "FRQ_U", "FRQ_DIFF", "FRQ_MEAN",
    "INFO", "log_SE", "neg_log10_HetPVal", "HetISqt_scaled",
    "log_HetDf", "log_Nca", "log_Nco", "log_Neff", "ngt",
    "direction_ratio",
]
NUM_NUMERIC = len(NUMERIC_FEATURES)

CHR_VOCAB = 23
CHR_EMBED_DIM = 8
ALLELE_VOCAB = 5
ALLELE_EMBED_DIM = 4
TOTAL_INPUT_DIM = NUM_NUMERIC + CHR_EMBED_DIM + ALLELE_EMBED_DIM * 2

MIN_INFO = 0.3
MIN_MAF = 0.001
MAX_MAF = 0.999
NORM_SAMPLE_FRAC = 0.01

HIDDEN_DIMS = [256, 128, 64]
DROPOUT = 0.2
HEAD_HIDDEN = 32

BATCH_SIZE = 8192
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
WARMUP_EPOCHS = 2

TRAIN_CHRS = list(range(1, 21))
VAL_CHRS = [21]
TEST_CHRS = [22]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()
