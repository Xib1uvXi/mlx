from dataclasses import dataclass
import pandas as pd
import pandas_ta as ta

from ml.feature_engineering.factors.interface import FactorInterface


@dataclass
class RSISetting:
    period: int = 14


class RSI(FactorInterface):
    def __init__(self, setting: RSISetting):
        self.setting = setting
        super().__init__(id=f"RSI_{self.setting.period}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi = ta.rsi(df["close"], length=self.setting.period)

        assert self.stationary(rsi), "RSI is not stationary"

        df[self.id()] = rsi
        return df


@dataclass
class OBVMADSetting:
    fast: int = 12
    slow: int = 26


class OBVMAD(FactorInterface):
    def __init__(self, setting: OBVMADSetting):
        self.setting = setting
        super().__init__(id=f"OBVMAD_f{self.setting.fast}_s{self.setting.slow}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        obv = ta.obv(df["close"], df["volume"])

        obv_fema = ta.ema(obv, length=self.setting.fast)
        obv_sema = ta.ema(obv, length=self.setting.slow)

        mad = ((obv_fema - obv_sema) / obv_sema) * 100

        assert self.stationary(mad), "OBVMAD is not stationary"

        df[self.id()] = mad
        return df
