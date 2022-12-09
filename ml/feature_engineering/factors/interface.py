from abc import ABC, abstractmethod
from dataclasses import dataclass
from arch.unitroot import ADF
import pandas as pd


@dataclass
class ADFSetting:
    """
    ADFSetting is a dataclass that holds the settings for the ADF test.
    """

    trend: str = "c"
    pvalue: float = 0.05
    show: bool = False


class FactorInterface(ABC):
    _id: str

    def __init__(self, id: str, setting: ADFSetting = ADFSetting()):
        self._id = id
        self.adf_setting = setting

    def stationary(self, series: pd.Series) -> bool:
        """
        stationary checks if the series is stationary.
        """

        adf = ADF(series.dropna(), trend=self.adf_setting.trend)

        result = adf.pvalue < self.adf_setting.pvalue

        if (not result) and self.adf_setting.show:
            print("The process contains a unit root")
            print(series.name, adf.summary().as_text())

        return result

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        transform transforms the data.
        """

        ...

    def id(self) -> str:
        """
        id returns the id of the factor.
        """

        return self._id
