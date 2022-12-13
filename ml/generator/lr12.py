from dataclasses import dataclass
from ml.generator.protocol import BaseGenerator, GeneratorSetting
import pandas as pd
import numpy as np
import pandas_ta as ta


@dataclass
class V1Setting(GeneratorSetting):
    n: int = 1
    ev: float = 0


class V1(BaseGenerator):
    """
    The V1 generator.
    """

    def __init__(self, setting: V1Setting = V1Setting()) -> None:
        super(V1, self).__init__(setting=setting)
        self.setting = setting
        self.df = self.ohlcv.copy()
        self.y()

    def y(self) -> None:
        self.df["log_close"] = np.log(self.df["close"])
        self.df["n_log_return"] = ta.log_return(self.df["close"], length=self.setting.n).shift(-self.setting.n)
        self.df["n_close_roc"] = ta.roc(self.df["close"], length=self.setting.n).shift(-self.setting.n)
        self.df["cumu_log_return"] = ta.log_return(self.df["close"], cumulative=True)

        self.df["y"] = self.df["n_log_return"].apply(lambda x: 1 if x > self.setting.ev else 0)

    def data(self) -> pd.DataFrame:
        return self.df

    def target(self) -> pd.Series:
        return self.df["y"]
