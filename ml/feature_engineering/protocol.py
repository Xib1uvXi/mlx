from typing import Protocol
import pandas as pd


class Factor(Protocol):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def id(self) -> str:
        ...
