import asyncio,os
from dataclasses import dataclass
from pathlib import Path

from fastcore.parallel import parallel_async
from fastcore.script import call_parse,is_cli
import pyarrow.parquet as pq

from fastforest.datasets import beyond_manifest

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

def _valid_parquet(path):
    if not path.exists() or not path.stat().st_size: return False
    try: pq.ParquetFile(path); return True
    except Exception: return False

def _download(task, data_home):
    from data_foundry.collections import BEYOND_ARENA
    cache = Path(data_home)/"data_foundry"
    container = BEYOND_ARENA.get_dataset(task["uuid"], cache_dir=str(cache), load_dataset=False)
    path = container.loaded_from_path/"dataset.parquet"
    downloaded = not _valid_parquet(path)
    if downloaded:
        container = BEYOND_ARENA.get_dataset(task["uuid"], cache_dir=str(cache), load_dataset=False, force_download=True)
        path = container.loaded_from_path/"dataset.parquet"
    if not _valid_parquet(path): raise ValueError("downloaded container has no valid dataset.parquet")
    return task["dataset_name"],downloaded,path

@dataclass(frozen=True)
class DownloadResults:
    cached:tuple
    downloaded:tuple
    failed:tuple

    def __repr__(self): return f"{len(self.cached)} cached · {len(self.downloaded)} downloaded · {len(self.failed)} failed"

async def download_tasks(tasks, data_home=".data/meta_benchmark", workers=4, timeout=300):
    "Download missing BeyondArena parquet payloads, retaining valid cached containers."
    tasks = list(tasks)
    results = await parallel_async(_download, tasks, data_home, n_workers=workers, timeout=timeout, return_exceptions=True)
    cached,downloaded,failed = [],[],[]
    for task,result in zip(tasks,results):
        if isinstance(result,Exception): failed.append((task["dataset_name"],f"{type(result).__name__}: {result}"))
        else: (downloaded if result[1] else cached).append(result[0])
    return DownloadResults(tuple(cached),tuple(downloaded),tuple(failed))

@call_parse
def main(
    metadata_csv:str="meta/meta_benchmark/beyondarena_metadata.csv", # Cached upstream task metadata
    data_home:str=".data/meta_benchmark", # Download cache
    task_type:str=None, # Optional random, grouped, or temporal selection
    task_names:str=None, # Optional comma-separated dataset names
    workers:int=4, # Concurrent container downloads
    timeout:int=300, # Maximum seconds per container
):
    "Download and verify BeyondArena dataset payloads independently of benchmarking."
    tasks = beyond_manifest(metadata_csv, include_text=True)
    if task_type is not None: tasks = tasks[tasks.task_type == task_type]
    if task_names:
        selected = {name.strip() for name in task_names.split(",") if name.strip()}
        tasks = tasks[tasks.dataset_name.isin(selected)]
    result = asyncio.run(download_tasks(tasks.to_dict("records"), data_home, workers, timeout))
    if is_cli(): print(result)
    else: return result

if __name__ == "__main__": main()
