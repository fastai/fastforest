import os,shutil,sys
from pathlib import Path

names = ("fastforest-fit", "fastforest-predict", "fastforest-compile", "fastforest-convert", "viewcsv")
root = Path(__file__).resolve().parents[1]
profile = sys.argv[1] if len(sys.argv) > 1 else "release"
source,destination = root/"target"/profile,root/"target"/"wheel-data"/"scripts"
destination.mkdir(parents=True, exist_ok=True)
suffix = ".exe" if os.name == "nt" else ""
for name in names: shutil.copy2(source/f"{name}{suffix}", destination/f"{name}{suffix}")
