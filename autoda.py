# autoda/autoda.py

from __future__ import annotations
import io, os, json, re, math, uuid, warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

import duckdb
warnings.filterwarnings("ignore")

class AutoDA:
    """
    Auto Data Analysis engine: 
    - loads, profiles, cleans, trains models, recommends charts, executes NLQ, generates reports.
    """

    def __init__(self, workdir: str = "assets"):
        self.workdir = Path(workdir)
        self.workdir.mkdir(exist_ok=True, parents=True)
        self.session_id = uuid.uuid4().hex[:8]
        self.df: Optional[pd.DataFrame] = None
        self.target: Optional[str] = None
        self.task: Optional[str] = None  # classification or regression
        self.report_steps: List[str] = []
        self.info: Dict = {}
        self.model = None
        self.pipeline: Optional[Pipeline] = None
        self.X_cols: List[str] = []
        self.y_name: Optional[str] = None
        self.viz_recos: List[Dict] = []
        self.fe_suggestions: List[str] = []

    # Data loading and profiling
    def load(self, file_bytes: bytes, filename: str) -> None:
        if filename.lower().endswith((".xlsx", ".xls")):
            self.df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            self.df = pd.read_csv(io.BytesIO(file_bytes))
        self.report_steps.append(f"Loaded dataset **{filename}** with shape {self.df.shape}.")

    def profile(self) -> Dict:
        types = self._infer_types()
        desc = self.df.describe(include="all", datetime_is_numeric=True).transpose().fillna("")
        profile = {
            "rows": int(self.df.shape[0]),
            "cols": int(self.df.shape[1]),
            "types": types,
            "missing_pct": self.df.isna().mean().round(3).to_dict(),
            "preview": self.df.head(5).to_dict(orient="records"),
        }
        self.info["profile"] = profile
        self.report_steps.append("Profiled data.")
        return profile

    def _infer_types(self) -> Dict[str, str]:
        types = {}
        for c in self.df.columns:
            s = self.df[c]
            if pd.api.types.is_numeric_dtype(s):
                types[c] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(s):
                types[c] = "datetime"
            elif s.nunique(dropna=True) <= 50:
                types[c] = "categorical"
            else:
                types[c] = "text"
        return types

    # Task detection and cleaning
    def set_target(self, target: str) -> None:
        assert self.df is not None
        if target not in self.df.columns:
            raise ValueError(f"Target '{target}' not found.")
        self.target = target
        y = self.df[target]
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > max(15, int(0.02*len(y))):
            self.task = "regression"
        else:
            self.task = "classification"
        self.report_steps.append(f"Selected target **{target}**; detected task: **{self.task}**.")

    def suggest_features(self) -> List[str]:
        types = self._infer_types()
        numeric_cols = [c for c,t in types.items() if t=="numeric" and c != self.target]
        cat_cols = [c for c,t in types.items() if t=="categorical" and c != self.target]

        sugs = []
        if cat_cols:
            sugs.append("Frequency encode high-cardinality categoricals.")
        if numeric_cols:
            sugs.append("Add pairwise interactions (A*B) for top-correlated numeric features.")
            sugs.append("Bin skewed numerics into quantiles (qcut) for tree models.")
        if any(t=="datetime" for t in types.values()):
            sugs.append("Expand datetimes into year/month/dayofweek and consider seasonal effects.")
        self.fe_suggestions = sugs
        self.report_steps.append("Generated feature engineering suggestions.")
        return sugs

    def _datetime_expand(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c, t in self._infer_types().items():
            if t == "datetime":
                s = pd.to_datetime(out[c], errors="coerce")
                out[f"{c}_year"]  = s.dt.year
                out[f"{c}_month"] = s.dt.month
                out[f"{c}_dow"]   = s.dt.dayofweek
        return out

    # Save artifacts and report
    def save_artifacts(self) -> Dict:
        model_path = self.workdir / f"model_{self.session_id}.pkl"
        report_path = self.workdir / f"report_{self.session_id}.md"
        info_path = self.workdir / f"info_{self.session_id}.json"

        import joblib
        joblib.dump(self.pipeline, model_path)

        # write report
        md = self._render_report_md()
        report_path.write_text(md, encoding="utf-8")

        # info json
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(self.info, f, indent=2)

        return {"model": str(model_path), "report": str(report_path), "info": str(info_path)}

    def _render_report_md(self) -> str:
        lines = ["# AutoDA Step-by-Step Report", ""]
        for i, step in enumerate(self.report_steps, 1):
            lines.append(f"{i}. {step}")
        return "
".join(lines)
