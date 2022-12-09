from dataclasses import dataclass
import pandas as pd
import pandas_ta as ta
import talib

from ml.feature_engineering.factors.interface import FactorInterface


@dataclass
class MADSetting:
    fast: int = 12
    slow: int = 26


class MAD(FactorInterface):
    def __init__(self, setting: MADSetting):
        self.setting = setting
        super().__init__(id=f"MAD_f{self.setting.fast}_s{self.setting.slow}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        sema = ta.sma(df["close"], length=self.setting.fast)
        fema = ta.sma(df["close"], length=self.setting.slow)

        mad = ((fema - sema) / sema) * 100

        assert self.stationary(mad), "MAD is not stationary"

        df[self.id()] = mad
        return df


@dataclass
class BLGSetting:
    period: int = 12


class BLG(FactorInterface):
    def __init__(self, setting: BLGSetting):
        self.setting = setting
        super().__init__(id=f"BLG_{self.setting.period}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        blg = (df["close"] - ta.sma(df["close"], self.setting.period)) / ta.stdev(
            df["close"], length=self.setting.period
        )

        assert self.stationary(blg), "BLG is not stationary"

        df[self.id()] = blg
        return df


@dataclass
class UUPSetting:
    period: int = 12


class UUP(FactorInterface):
    def __init__(self, setting: UUPSetting):
        self.setting = setting
        super().__init__(id=f"UUP_{self.setting.period}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        roc = ta.roc(df["close"], length=1)

        uup = talib.SUM(
            (roc.apply(lambda x: x if x > 0 else 0) * df["volume"]), timeperiod=self.setting.period
        ) / talib.SUM((roc.apply(lambda x: x if x > 0 else abs(x)) * df["volume"]), timeperiod=self.setting.period)

        assert self.stationary(uup), "UUP is not stationary"

        df[self.id()] = uup
        return df
