"""Build the paper's growth extension from the same code as notebooks 14a--14c."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/src"))
from brainsmodels.growth_lessons import write_paper_assets

if __name__ == "__main__":
    print(write_paper_assets(ROOT))
