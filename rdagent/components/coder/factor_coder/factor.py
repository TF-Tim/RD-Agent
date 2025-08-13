from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from filelock import FileLock

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.components.coder.CoSTEER.task import CoSTEERTask
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.core.exception import CodeFormatError, CustomRuntimeError, NoOutputError
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.core.utils import cache_with_pickle
from rdagent.oai.llm_utils import md5_hash


class FactorTask(CoSTEERTask):
    # TODO: generalized the attributes into the Task
    # - factor_* -> *
    def __init__(
        self,
        factor_name,
        factor_description,
        factor_formulation,
        *args,
        variables: dict = {},
        resource: str = None,
        factor_implementation: bool = False,
        **kwargs,
    ) -> None:
        self.factor_name = (
            factor_name  # TODO: remove it in the later version. Keep it only for pickle version compatibility
        )
        self.factor_formulation = factor_formulation
        self.variables = variables
        self.factor_resources = resource
        self.factor_implementation = factor_implementation
        super().__init__(name=factor_name, description=factor_description, *args, **kwargs)

    @property
    def factor_description(self):
        """for compatibility"""
        return self.description

    def get_task_information(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}"""

    def get_task_brief_information(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}"""

    def get_task_information_and_implementation_result(self):
        return {
            "factor_name": self.factor_name,
            "factor_description": self.factor_description,
            "factor_formulation": self.factor_formulation,
            "variables": str(self.variables),
            "factor_implementation": str(self.factor_implementation),
        }

    @staticmethod
    def from_dict(dict):
        return FactorTask(**dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.factor_name}]>"


class FactorFBWorkspace(FBWorkspace):
    """
    This class is used to implement a factor by writing the code to a file.
    Input data and output factor value are also written to files.
    """

    # TODO: (Xiao) think raising errors may get better information for processing
    FB_EXEC_SUCCESS = "Execution succeeded without error."
    FB_CODE_NOT_SET = "code is not set."
    FB_EXECUTION_SUCCEEDED = "Execution succeeded without error."
    FB_OUTPUT_FILE_NOT_FOUND = "\nExpected output file not found."
    FB_OUTPUT_FILE_FOUND = "\nExpected output file found."

    def __init__(
        self,
        *args,
        raise_exception: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.raise_exception = raise_exception

    def hash_func(self, data_type: str = "Debug") -> str:
        return (
            md5_hash(data_type + self.file_dict["factor.py"])
            if ("factor.py" in self.file_dict and not self.raise_exception)
            else None
        )

    @cache_with_pickle(hash_func)
    def execute(self, data_type: str = "Debug") -> Tuple[str, pd.DataFrame]:
        """
        execute the implementation and get the factor value by the following steps:
        1. make the directory in workspace path
        2. write the code to the file in the workspace path
        3. link all the source data to the workspace path folder
        if call_factor_py is True:
            4. execute the code
        else:
            4. generate a script from template to import the factor.py dump get the factor value to result.h5
        5. read the factor value from the output file in the workspace path folder
        returns the execution feedback as a string and the factor value as a pandas dataframe

        Regarding the cache mechanism:
        1. We will store the function's return value to ensure it behaves as expected.
        - The cached information will include a tuple with the following: (execution_feedback, executed_factor_value_dataframe, Optional[Exception])
        """
        self.before_execute()
        if self.file_dict is None or "factor.py" not in self.file_dict:
            if self.raise_exception:
                raise CodeFormatError(self.FB_CODE_NOT_SET)
            else:
                return self.FB_CODE_NOT_SET, None

        with FileLock(self.workspace_path / "execution.lock"):
            if self.target_task.version == 1:
                source_data_path = (
                    Path(FACTOR_COSTEER_SETTINGS.data_folder_debug)
                    if data_type == "Debug"  # FIXME: (yx) don't think we should use a debug tag for this.
                    else Path(FACTOR_COSTEER_SETTINGS.data_folder)
                )
            elif self.target_task.version == 2:
                # TODO you can change the name of the data folder for a better understanding
                source_data_path = Path(KAGGLE_IMPLEMENT_SETTING.local_data_path) / KAGGLE_IMPLEMENT_SETTING.competition
            else:
                source_data_path = Path(FACTOR_COSTEER_SETTINGS.data_folder)

            source_data_path.mkdir(exist_ok=True, parents=True)
            code_path = self.workspace_path / "factor.py"

            self.link_all_files_in_folder_to_workspace(source_data_path, self.workspace_path)
            # --- begin: auto-fix common issues in generated factor.py ---
            def _auto_fix_factor_py(code_path):
                import re
                from pathlib import Path
                try:
                    t = Path(code_path).read_text()
                except Exception:
                    return
                # 1) Remove invalid MultiIndex reindex like: result.index = df.index[result.index]
                t2 = re.sub(
                    r"^\s*result\s*\.\s*index\s*=\s*df\s*\.\s*index\s*\[\s*result\s*\.\s*index\s*\]\s*$",
                    "# SAFE-EDIT: removed invalid reindex of MultiIndex",
                    t,
                    flags=re.M
                )
                # 2) Ensure .to_hdf('result.h5', key='data', mode='w')
                def _fix_to_hdf(m):
                    call = m.group(0)
                    if "key=" not in call:
                        call = call[:-1] + ", key=\"data\")"
                    if "mode=" not in call:
                        call = call[:-1] + ", mode=\"w\")"
                    return call
                t3 = re.sub(r"\.to_hdf\(\s*['\"]result\.h5['\"][^)]*\)", _fix_to_hdf, t2)
                if t3 != t:
                    Path(code_path).write_text(t3)

            code_path = self.workspace_path / "factor.py"
            _auto_fix_factor_py(code_path)
            # --- end: auto-fix common issues in generated factor.py ---
            
            # --- begin: auto-group time-series ops by instrument ---
            def _auto_group_factor_py(code_path):
                import re
                from pathlib import Path
                try:
                    t = Path(code_path).read_text()
                except Exception:
                    return
                # 1) df['col'].(shift|diff|pct_change)(...)
                t1 = re.sub(
                    r"(df\[['\"](?P<col>\w+)['\"]\]\s*\.\s*(?P<op>shift|diff|pct_change)\s*\()",
                    r"df.groupby(level='instrument')['\g<col>'].\g<op>(",
                    t,
                )
                # 2) df.col.(shift|diff|pct_change)(...)
                t2 = re.sub(
                    r"(df\.(?P<col>\w+)\s*\.\s*(?P<op>shift|diff|pct_change)\s*\()",
                    r"df.groupby(level='instrument')['\g<col>'].\g<op>(",
                    t1,
                )
                # 3) df['col'].rolling(win).agg(...)
                def _fix_roll_agg(m):
                    col = m.group('col'); win = m.group('win'); agg = m.group('agg'); args = m.group('args')
                    return f"(df.groupby(level='instrument')['{col}'].rolling({win}).{agg}({args}).reset_index(level=0, drop=True))"
                t3 = re.sub(
                    r"df\[['\"](?P<col>\w+)['\"]\]\s*\.\s*rolling\((?P<win>[^)]*)\)\s*\.\s*(?P<agg>mean|sum|max|min|std|var|median|apply)\((?P<args>[^)]*)\)",
                    _fix_roll_agg,
                    t2,
                )
                if t3 != t:
                    Path(code_path).write_text(t3)

            code_path = self.workspace_path / "factor.py"
            _auto_group_factor_py(code_path)
            # --- end: auto-group time-series ops by instrument ---



            # --- begin: prepare a local daily_pv.h5 we control (no writes to shared ~/.qlib files)
            def _first_key(h5_path: Path):
                import pandas as pd
                try:
                    with pd.HDFStore(h5_path, "r") as store:
                        keys = store.keys()
                        if not keys:
                            return None
                        return "/data" if "/data" in keys else keys[0]
                except Exception:
                    return None

            def _make_local_pv(src: Path, dst: Path):
                import pandas as pd
                if not src.exists():
                    return False, "src-missing"
                key = _first_key(src)
                if not key:
                    return False, "no-key"

                df = pd.read_hdf(src, key=key.strip("/"))

                # Ensure MultiIndex ['datetime','instrument']
                if not isinstance(df.index, pd.MultiIndex) or set(df.index.names) != {"datetime", "instrument"}:
                    if "datetime" not in df.columns or "instrument" not in df.columns:
                        return False, "bad-index"
                    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                    df = df.set_index(["datetime", "instrument"]).sort_index()

                # Add unprefixed OHLCV columns if only $prefixed ones exist
                mapping = {"$open": "open", "$high": "high", "$low": "low", "$close": "close", "$volume": "volume"}
                for s, t in mapping.items():
                    if s in df.columns and t not in df.columns:
                        df[t] = df[s]

                # Write to a real local file (not a symlink) so factors can read 'daily_pv.h5'
                if dst.exists() or dst.is_symlink():
                    try:
                        dst.unlink()
                    except Exception:
                        pass
                df.to_hdf(dst, key="data", mode="w")
                return True, f"rows={len(df)} cols={len(df.columns)}"

            p_debug = self.workspace_path / "daily_pv_debug.h5"
            p_local = self.workspace_path / "daily_pv.h5"
            ok, msg = (False, "not-run")
            if p_debug.exists():
                ok, msg = _make_local_pv(p_debug, p_local)

            # Optional breadcrumb for troubleshooting
            try:
                (self.workspace_path / "_pv_prep.txt").write_text(f"prepared={ok} {msg}\n")
            except Exception:
                pass
            # --- end: prepare a local daily_pv.h5
           
            execution_feedback = self.FB_EXECUTION_SUCCEEDED
            execution_success = False
            execution_error = None  # kept for compatibility with cached payloads

            if self.target_task.version == 1:
                execution_code_path = code_path
            elif self.target_task.version == 2:
                execution_code_path = self.workspace_path / f"{uuid.uuid4()}.py"
                execution_code_path.write_text((Path(__file__).parent / "factor_execution_template.txt").read_text())
            else:
                execution_code_path = code_path

            try:
                subprocess.check_output(
                    f"{FACTOR_COSTEER_SETTINGS.python_bin} {execution_code_path}",
                    shell=True,
                    cwd=self.workspace_path,
                    stderr=subprocess.STDOUT,
                    timeout=FACTOR_COSTEER_SETTINGS.file_based_execution_timeout,
                )
                execution_success = True
            except subprocess.CalledProcessError as e:
                import site

                execution_feedback = (
                    e.output.decode()
                    .replace(str(execution_code_path.parent.absolute()), r"/path/to")
                    .replace(str(site.getsitepackages()[0]), r"/path/to/site-packages")
                )
                if len(execution_feedback) > 2000:
                    execution_feedback = (
                        execution_feedback[:1000] + "....hidden long error message...." + execution_feedback[-1000:]
                    )
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback)
                else:
                    execution_error = CustomRuntimeError(execution_feedback)
            except subprocess.TimeoutExpired:
                execution_feedback += (
                    f"Execution timeout error and the timeout is set to "
                    f"{FACTOR_COSTEER_SETTINGS.file_based_execution_timeout} seconds."
                )
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback)
                else:
                    execution_error = CustomRuntimeError(execution_feedback)

            # ---- Read & normalize result.h5 (key='data'), ensure MultiIndex and single numeric column
            workspace_output_file_path = self.workspace_path / "result.h5"
            if workspace_output_file_path.exists() and execution_success:
                try:
                    df = pd.read_hdf(workspace_output_file_path, key="data")

                    def _normalize(df_in: pd.DataFrame) -> pd.DataFrame:
                        df2 = df_in

                        # Ensure MultiIndex ['datetime','instrument']
                        bad_index = (
                            not isinstance(df2.index, pd.MultiIndex)
                            or df2.index.nlevels != 2
                            or set(df2.index.names) != {"datetime", "instrument"}
                        )
                        if bad_index:
                            df2 = df2.reset_index()

                            # Try to find datetime column if it's named differently
                            if "datetime" not in df2.columns:
                                for cand in ["date", "Date", "level_0"]:
                                    if cand in df2.columns:
                                        df2 = df2.rename(columns={cand: "datetime"})
                                        break

                            # Try to find instrument column if it's named differently
                            if "instrument" not in df2.columns:
                                for cand in ["ticker", "symbol", "Instrument", "level_1", "instrument_1", "secid"]:
                                    if cand in df2.columns:
                                        df2 = df2.rename(columns={cand: "instrument"})
                                        break

                            if "datetime" not in df2.columns or "instrument" not in df2.columns:
                                raise ValueError("result.h5 has unexpected index/columns; cannot normalize.")

                            df2["datetime"] = pd.to_datetime(df2["datetime"], errors="coerce")
                            df2 = df2.set_index(["datetime", "instrument"]).sort_index()

                        # Reorder/fix names if needed
                        if list(df2.index.names) != ["datetime", "instrument"]:
                            try:
                                df2 = df2.reorder_levels(["datetime", "instrument"]).sort_index()
                            except Exception:
                                pass
                        df2.index.names = ["datetime", "instrument"]

                        # Keep a single numeric column
                        if df2.shape[1] != 1:
                            numeric_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
                            if not numeric_cols:
                                raise ValueError("result.h5 contains no numeric factor column.")
                            df2 = df2[[numeric_cols[0]]]

                        return df2

                    executed_factor_value_dataframe = _normalize(df)
                    execution_feedback += self.FB_OUTPUT_FILE_FOUND
                except Exception as e:
                    execution_feedback += f"Error found when reading hdf file: {e}"[:1000]
                    executed_factor_value_dataframe = None
            else:
                execution_feedback += self.FB_OUTPUT_FILE_NOT_FOUND
                executed_factor_value_dataframe = None
                if self.raise_exception:
                    raise NoOutputError(execution_feedback)
                else:
                    execution_error = NoOutputError(execution_feedback)

        return execution_feedback, executed_factor_value_dataframe

    def __str__(self) -> str:
        # NOTE:
        # If the code cache works, the workspace will be None.
        return f"File Factor[{self.target_task.factor_name}]: {self.workspace_path}"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_folder(task: FactorTask, path: Union[str, Path], **kwargs):
        path = Path(path)
        code_dict = {}
        for file_path in path.iterdir():
            if file_path.suffix == ".py":
                code_dict[file_path.name] = file_path.read_text()
        return FactorFBWorkspace(target_task=task, code_dict=code_dict, **kwargs)


FactorExperiment = Experiment
FeatureExperiment = Experiment
