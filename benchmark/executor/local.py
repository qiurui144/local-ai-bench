"""LocalExecutor — 封装现有本机执行行为（向后兼容）。"""
from pathlib import Path


class LocalExecutor:
    def run_benchmark(self, model_name: str, extra_args: list[str] = ()) -> Path:
        """直接在本进程调用 run_benchmark 逻辑；返回报告目录 Path。"""
        import run_benchmark as rb
        import sys
        old_argv = sys.argv
        try:
            sys.argv = ["run_benchmark.py", "--model", model_name, *extra_args]
            output_dir = rb.main()
            return Path(output_dir) if output_dir else Path("output/reports")
        finally:
            sys.argv = old_argv
