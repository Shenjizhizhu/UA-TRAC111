import os
import torch

SEED = 42

ROOT_PATH = os.path.join('/', 'home', 'adduser', 'Shenji', 'UA-TRAC')
IMG_ROOT = os.path.join(ROOT_PATH, 'Insight-MVT_Annotation_Train')
ORIGINAL_GT_PATH = os.path.join(ROOT_PATH, 'train_gt.txt')
OUTPUT_IMG_DIR = os.path.join(ROOT_PATH, 'outputting_imgs')
NEW_GT_DIR = os.path.join(ROOT_PATH, 'new_gt_files')

TEST_RATIO = 0.1
VAL_RATIO = 0.1
EPOCHS = 5
BATCH_SIZE = 8
MAX_BOXES = 30
IMG_SIZE = 640
NUM_CLASSES = 1
MAX_SAMPLES = None

LR = 0.2
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
ACCUMULATE_STEPS = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')