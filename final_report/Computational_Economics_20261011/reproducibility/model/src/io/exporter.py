"""数据导出 —— HDF5存储和学术格式输出."""

import os
import json
import numpy as np
import pandas as pd


class ResultExporter:
    """模拟结果导出器."""

    @staticmethod
    def to_hdf5(results: list, filepath: str):
        """将批量模拟结果导出为HDF5."""
        import h5py

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        with h5py.File(filepath, "w") as f:
            # 元数据
            f.attrs["total_runs"] = len(results)
            f.attrs["timestamp"] = str(pd.Timestamp.now())

            # 汇总表
            summaries = [r["summary"] for r in results]
            summ_df = pd.DataFrame(summaries)
            for col in summ_df.columns:
                if summ_df[col].dtype == object:
                    # 字符串列存储为属性
                    pass
                else:
                    f.create_dataset(f"summary/{col}", data=summ_df[col].values)

            # 存储每个run的断裂类型（字符串需特殊处理）
            dt = h5py.string_dtype()
            fracture_types_data = [
                ",".join(s.get("fracture_types", [])) for s in summaries
            ]
            f.create_dataset("summary/fracture_types_str", data=fracture_types_data, dtype=dt)
            f.create_dataset("summary/terminal_fracture_str",
                             data=[str(s.get("terminal_fracture", "")) for s in summaries], dtype=dt)
            f.create_dataset("summary/deind_type_final_str",
                             data=[str(s.get("deind_type_final", "")) for s in summaries], dtype=dt)

            # 时间序列（仅保存前N个以控制文件大小）
            max_ts = min(len(results), 100)
            ts_group = f.create_group("time_series")
            for i in range(max_ts):
                if "dataframe" in results[i]:
                    grp = ts_group.create_group(f"run_{i:04d}")
                    df = results[i]["dataframe"]
                    for col in df.columns:
                        grp.create_dataset(col, data=df[col].values)

            # 参数
            if results and "params" in results[0]:
                params_df = pd.DataFrame([r.get("params", {}) for r in results])
                param_group = f.create_group("params")
                for col in params_df.columns:
                    if params_df[col].dtype != object:
                        param_group.create_dataset(col, data=params_df[col].values)

        print(f"Results exported to {filepath}")

    @staticmethod
    def to_csv_summary(results: list, filepath: str):
        """导出汇总CSV."""
        summaries = [r["summary"] for r in results]
        df = pd.DataFrame(summaries)
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Summary exported to {filepath}")

    @staticmethod
    def to_json_report(result: dict, filepath: str):
        """导出单次模拟的JSON报告."""
        report = {
            "summary": result["summary"],
            "fracture_events": [],
        }
        if "dataframe" in result:
            df = result["dataframe"]
            report["final_values"] = {col: float(df[col].iloc[-1]) for col in df.columns
                                      if not df[col].isna().all()}

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"Report exported to {filepath}")

    @staticmethod
    def to_excel(results: list, filepath: str):
        """导出为Excel（含多个sheet）."""
        summaries = [r["summary"] for r in results]
        summ_df = pd.DataFrame(summaries)

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            summ_df.to_excel(writer, sheet_name="汇总", index=False)

            # 选几个代表性模拟的时间序列
            for i in range(min(5, len(results))):
                if "dataframe" in results[i]:
                    results[i]["dataframe"].to_excel(
                        writer, sheet_name=f"模拟{i+1}_时间序列", index=False
                    )

        print(f"Excel report exported to {filepath}")
