from dataclasses import dataclass
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd


@dataclass
class ExtractorSetting:
    id: str
    roll_max_timeshift: int
    roll_min_timeshift: int
    n_jobs: int = 8


class Extractor:
    def __init__(self, setting: ExtractorSetting) -> None:
        self.setting = setting

    def extract(self, data: dict) -> pd.DataFrame:
        melted = self.__melt(data)
        rolled = self.__roll_dataframe(melted)
        X = self.__extract_features(rolled)
        return X

    def select_features(self, y: pd.Series) -> pd.DataFrame:
        y = y[y.index.isin(self.X.index)]
        X = self.X[self.X.index.isin(y.index)]

        X_filtered = select_features(X, y)
        self.X_filtered = X_filtered
        return X_filtered

    def __melt(self, data: dict) -> pd.DataFrame:
        melted = pd.DataFrame(data)
        melted["opentime"] = melted.index
        melted["column_id"] = self.setting.id

        self.melted = melted
        return melted

    def __roll_dataframe(self, melted: pd.DataFrame) -> pd.DataFrame:
        rolled = rolled = roll_time_series(
            melted,
            column_id="column_id",
            column_sort="opentime",
            column_kind=None,
            rolling_direction=1,
            max_timeshift=self.setting.roll_max_timeshift,
            min_timeshift=self.setting.roll_min_timeshift,
            show_warnings=False,
            n_jobs=self.setting.n_jobs,
        )

        self.rolled = rolled
        return rolled

    def __extract_features(self, rolled: pd.DataFrame) -> pd.DataFrame:
        X = extract_features(
            rolled.drop("column_id", axis=1),
            column_id="id",
            column_sort="opentime",
            impute_function=impute,
            show_warnings=False,
            n_jobs=self.setting.n_jobs,
        )

        X = X.set_index(X.index.map(lambda x: x[1]), drop=True)
        X.index.name = "open_time"

        self.X = X
        return X
