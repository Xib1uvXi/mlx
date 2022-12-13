from dataclasses import dataclass
import pandas as pd
import pandas_ta as ta

from ml.feature_engineering.factors.interface import FactorInterface


@dataclass
class RSISetting:
    period: int = 14


class RSI(FactorInterface):
    def __init__(self, setting: RSISetting = RSISetting()):
        self.setting = setting
        super().__init__(id=f"RSI_{self.setting.period}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rsi = ta.rsi(df["close"], length=self.setting.period)

        assert self.stationary(rsi), "RSI is not stationary"

        df[self.id()] = rsi
        return df


class OBV(FactorInterface):
    def __init__(self):
        super().__init__(id="OBV")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        obv = ta.obv(df["close"], df["volume"])
        df[self.id()] = obv
        return df
