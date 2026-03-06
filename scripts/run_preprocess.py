from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    runpy.run_module("movie_recommendation.pipelines.preprocess_sasrec", run_name="__main__")
