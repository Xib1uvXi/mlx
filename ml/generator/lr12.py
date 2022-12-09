from dataclasses import dataclass
from ml.generator.protocol import BaseGenerator
import pandas as pd
from datetime import datetime
from typing import Optional
import numpy as np
import pandas_ta as ta


@dataclass
class V1Setting:
    # symbol: str = "btcusdtperp"
    symbol: str = "btcusdt"
    interval: str = "1h"
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    n: int = 8
    ev: float = 0.01

    def tables(self) -> str:
        return f"bars_{self.symbol}_{self.interval}"

    def database(self) -> str:
        # TODO use config
        return "binance_data"


class V1(BaseGenerator):
    """
    The V1 generator.
    """

    def __init__(self, setting: V1Setting = V1Setting()) -> None:
        super(V1, self).__init__()
        self.setting = setting
        self.prep_ohlcv = self.load_db(
            self.setting.database(), self.setting.tables(), self.setting.start, self.setting.end
        )
        # FIXME bad practice
        self.spot_ohlcv = self.load_db(
            "binance_data", f"bars_btcusdt_{self.setting.interval}", self.setting.start, self.setting.end
        )

        self.df = self.prep_ohlcv.copy()
        self.df["spot_close"] = self.spot_ohlcv["close"]
        self.df["spot_volume"] = self.spot_ohlcv["volume"]
        self.y()

    def y(self) -> None:
        self.df["log_close"] = np.log(self.df["close"])
        self.df["n_log_return"] = ta.log_return(self.df["close"], length=self.setting.n).shift(-self.setting.n)
        self.df["n_close_roc"] = ta.roc(self.df["close"], length=self.setting.n).shift(-self.setting.n)
        self.df["cumu_log_return"] = ta.log_return(self.df["close"], cumulative=True)

        self.df["y"] = self.df["n_log_return"].apply(
            lambda x: 1 if x > self.setting.ev else -1 if x < -self.setting.ev else 0
        )

    def data(self) -> pd.DataFrame:
        return self.df

    def target(self) -> pd.Series:
        return self.df["y"]
