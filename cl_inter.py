from consistlearning.task import VanillaTrain
from cl import *

import torch as T
import numpy as np

from utils import progress_bar
from .cl import EvalProgressPerSample as EPSP, FixDataMemoryBatchClassification as FD, MetricClassification as MC



