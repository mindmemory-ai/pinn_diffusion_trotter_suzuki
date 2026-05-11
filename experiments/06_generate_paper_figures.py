"""Generate all paper figures (阶段 9-C: 一键生成脚本)."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pinn_trotter.visualization.figure_generator import main

if __name__ == "__main__":
    main()
