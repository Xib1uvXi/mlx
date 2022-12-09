import os
from ml.feature_engineering.protocol import Factor
import pandas as pd
from ml.feature_engineering.derivative_features_extractor import Extractor
import uuid
from pyarrow import feather


class MLXFeaturesFactory:
    derivative_extractor: Extractor

    def __init__(self):
        self.factors: list[Factor] = []

    def set_extractor(self, extractor: Extractor):
        self.derivative_extractor = extractor

    def add_factor(self, factors: list[Factor]):
        self.factors.extend(factors)

    def factor_extract(self, df: pd.DataFrame) -> pd.DataFrame:
        for factor in self.factors:
            df = factor.transform(df)
        return df

    def derivative_features(self, x_data: dict, y: pd.Series) -> pd.DataFrame:
        self.derivative_extractor.extract(x_data)
        return self.derivative_extractor.select_features(y)

    def save_derivative_features(self, features: pd.DataFrame) -> str:
        _uuid = str(uuid.uuid4())

        cache_file = "/tmp/" + _uuid + ".feather"

        feather.write_feather(features, cache_file)

        return _uuid

    def load_derivative_features_cache(self, _uuid: str) -> pd.DataFrame:
        cache_file = "/tmp/" + _uuid + ".feather"

        if not os.path.exists(cache_file):
            raise Exception("Cache file does not exist")

        return feather.read_feather(cache_file)
