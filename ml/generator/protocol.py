from abc import ABC, abstractmethod
from typing import Optional, Protocol
import myloginpath
from datetime import datetime
import os
from pyarrow import feather
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np


class Generator(Protocol):
    def data(self) -> pd.DataFrame:
        ...

    def target(self) -> pd.Series:
        ...


class BaseGenerator(ABC):
    def __init__(self) -> None:
        self.dbconf = myloginpath.parse("client")

    def load_db(
        self, database: str, tables: str, start: Optional[datetime] = None, end: Optional[datetime] = None
    ) -> pd.DataFrame:
        conn = f"mysql://{self.dbconf['user']}:{self.dbconf['password']}@{self.dbconf['host']}/{database}"

        cache_file = os.path.join(os.getenv("MXL_LD_CACHE_DIR", f"/tmp"), f"{database}_{tables}.feather")
        if os.path.exists(cache_file):
            return pd.read_feather(cache_file)

        sql = (
            "select b.datetime open_time, b.open_price open, b.high_price high, b.low_price low, b.close_price close, b.volume"
            f" from {tables} b where 1=1"
        )

        if start:
            sql += f" and b.datetime >= '{start}'"

        if end:
            sql += f" and b.datetime <= '{end}'"

        sql += " order by b.datetime asc"
        df = pd.read_sql(sql, conn, index_col=["open_time"])
        df = df.tz_localize("Asia/Hong_Kong")

        feather.write_feather(df, cache_file)
        return df

    def correlation(self, k1: pd.Series, k2: pd.Series) -> float:
        return np.corrcoef(k1.values, k2.values)[0, 1]

    def correlation_plot(self, df: pd.DataFrame, method: str = "all") -> None:
        fig = make_subplots(
            rows=1,
            cols=3 if method == "all" else 1,
            subplot_titles=(
                "皮尔逊积矩相关系数 (Pearson correlation coefficient)",
                "肯德尔等级相关系数 (Kendall rank correlation coefficient)",
                "斯皮尔曼等级相关系数 (Spearman's rank correlation coefficient)",
            )
            if method == "all"
            else ("皮尔逊积矩相关系数 (Pearson correlation coefficient)",)
            if method == "pearson"
            else ("肯德尔等级相关系数 (Kendall rank correlation coefficient)",)
            if method == "kendall"
            else ("斯皮尔曼等级相关系数 (Spearman's rank correlation coefficient)",),
        )

        def generate_heatmap(df: pd.DataFrame, col: int):
            fig.add_trace(
                go.Heatmap(
                    x=df.columns,
                    y=df.index,
                    z=np.array(df),
                    text=df.values,
                    texttemplate="%{text:.2f}",
                    colorscale=px.colors.diverging.RdBu,
                ),
                row=1,
                col=col,
            )

        if method == "all":
            generate_heatmap(df.corr(method="pearson"), 1)
            generate_heatmap(df.corr(method="kendall"), 2)
            generate_heatmap(df.corr(method="spearman"), 3)

        elif method == "pearson":
            generate_heatmap(df.corr(method="pearson"), 1)

        elif method == "kendall":
            generate_heatmap(df.corr(method="kendall"), 1)

        elif method == "spearman":
            generate_heatmap(df.corr(method="spearman"), 1)

        fig.update_layout(
            title="相关系数矩阵 (Correlation matrix)",
        )

        fig.show()

    @abstractmethod
    def data(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def target(self) -> pd.Series:
        ...

    @abstractmethod
    def y(self) -> None:
        ...
