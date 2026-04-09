"""Central configuration for PRS Neural Network training."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW_DIR = PROJECT_ROOT / "data.nosync" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data.nosync" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints.nosync"
LOG_DIR = PROJECT_ROOT / "logs.nosync"
NORM_STATS_PATH = DATA_PROCESSED_DIR / "norm_stats.json"

# ── Dataset ────────────────────────────────────────────────────────────
HF_DATASET = "OpenMed/pgc-schizophrenia"
HF_CONFIG = "scz2022"
DOWNLOAD_CHUNK_SIZE = 100_000  # rows per streaming chunk

# ── Feature engineering ────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "BP_norm", "FRQ_A", "FRQ_U", "FRQ_DIFF", "FRQ_MEAN",
    "INFO", "log_SE", "neg_log10_HetPVal", "HetISqt_scaled",
    "log_HetDf", "log_Nca", "log_Nco", "log_Neff", "ngt",
    "direction_ratio",
]
NUM_NUMERIC = len(NUMERIC_FEATURES)  # 15

# Categorical embedding dims
CHR_VOCAB = 23       # chr 1-22 + unknown
CHR_EMBED_DIM = 8
ALLELE_VOCAB = 5     # A, C, G, T, unknown
ALLELE_EMBED_DIM = 4

TOTAL_INPUT_DIM = NUM_NUMERIC + CHR_EMBED_DIM + ALLELE_EMBED_DIM * 2  # 31

# Filtering thresholds
MIN_INFO = 0.3
MIN_MAF = 0.001
MAX_MAF = 0.999

# Normalization sampling
NORM_SAMPLE_FRAC = 0.01

# ── Model ──────────────────────────────────────────────────────────────
HIDDEN_DIMS = [256, 128, 64]
DROPOUT = 0.2
HEAD_HIDDEN = 32

# ── Training ───────────────────────────────────────────────────────────
BATCH_SIZE = 8192
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
WARMUP_EPOCHS = 2

# Chromosome splits
TRAIN_CHRS = list(range(1, 21))   # chr 1-20
VAL_CHRS = [21]
TEST_CHRS = [22]

# ── Device ─────────────────────────────────────────────────────────────
def get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _lazy_device():
    """Lazy device resolution — only called when DEVICE is accessed at runtime."""
    import torch
    return get_device()


class _LazyDevice:
    """Proxy that defers torch import until attribute access."""
    _device = None

    def _resolve(self):
        if self._device is None:
            self.__class__._device = _lazy_device()
        return self._device

    def __getattr__(self, name):
        return getattr(self._resolve(), name)

    def __repr__(self):
        return repr(self._resolve())

    def __eq__(self, other):
        return self._resolve() == other

    def __hash__(self):
        return hash(self._resolve())


DEVICE = _LazyDevice()
