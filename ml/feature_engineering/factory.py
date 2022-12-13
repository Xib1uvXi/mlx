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

    def set_extractor(self, extractor: Extractor) -> "MLXFeaturesFactory":
        self.derivative_extractor = extractor

        return self

    def add_factor(self, factors: list[Factor]) -> "MLXFeaturesFactory":
        self.factors.extend(factors)

        return self

    def factor_extract(self, df: pd.DataFrame) -> pd.DataFrame:
        for factor in self.factors:
            df = factor.transform(df)
        return df

    def derivative_extract(self, df: pd.DataFrame, dtag: list[Factor], y: pd.Series) -> pd.DataFrame:
        df = df.dropna()
        dfe_dict: dict[str, pd.Series] = {}
        for tag in dtag:
            dfe_dict[tag.id()] = df[tag.id()]

        X = self.__derivative_features(dfe_dict, y)

        return X

    def __derivative_features(self, x_data: dict, y: pd.Series) -> pd.DataFrame:
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
