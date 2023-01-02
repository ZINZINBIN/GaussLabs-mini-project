import pandas as pd
from src.config import Config
from src.feature_extraction import time_features

config = Config()

data = pd.read_csv(config.DATA_PATH['ettm1'])

print(time_features(data))