from pathlib import Path

from ccl_science_data.common import oa_persistent

parent_dir = oa_persistent / "gender-class-base"
export_dir = Path(oa_persistent / "2024-09-openalex-paper-level-classification/")
step0_dir = export_dir / "step0"
step0_dir.mkdir(exist_ok=True, parents=True)
step1_dir = export_dir / "step1"
step1_dir.mkdir(exist_ok=True)
step2_dir = export_dir / "step2"
step2_dir.mkdir(exist_ok=True)
