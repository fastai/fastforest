import os,shutil
from pathlib import Path

names = ("fastforest-fit", "fastforest-predict", "fastforest-compile", "fastforest-convert", "viewcsv")
root = Path(__file__).resolve().parents[1]
source,destination = root/"target"/"release",root/"target"/"wheel-data"/"scripts"
destination.mkdir(parents=True, exist_ok=True)
suffix = ".exe" if os.name == "nt" else ""
for name in names: shutil.copy2(source/f"{name}{suffix}", destination/f"{name}{suffix}")
