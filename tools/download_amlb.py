import asyncio,csv,json,re,urllib.parse,urllib.request
from pathlib import Path

from fastcore.parallel import parallel_async
from fastcore.script import call_parse
import pyarrow.parquet as pq

GITLAB_API = "https://gitlab.com/api/v4/groups/data%2Fd%2Fopenml/projects"

def _json(url):
    with urllib.request.urlopen(url, timeout=30) as response: return json.load(response)

def _qualities(project):
    url = f"{project['web_url']}/-/raw/master/dataset/qualities.json"
    values = _json(url)["data_qualities"]["quality"]
    return {item["name"]:item["value"] for item in values}

def _number(value):
    try: return int(float(value))
    except (TypeError,ValueError): return None

def _resolve(item):
    if item.get("openml_id"): return int(float(item["openml_id"]))
    query = urllib.parse.urlencode(dict(search=item["name"], per_page=100, simple="true"))
    projects = _json(f"{GITLAB_API}?{query}")
    matches = []
    for project in projects:
        try: qualities = _qualities(project)
        except Exception: continue
        if _number(qualities.get("NumberOfInstances")) != int(item["rows"]): continue
        if _number(qualities.get("NumberOfFeatures")) != int(item["features"]): continue
        classes = _number(qualities.get("NumberOfClasses"))
        if item["task"] == "classification" and classes != int(item["classes"]): continue
        if item["task"] == "regression" and classes not in (None,0): continue
        matches.append(int(project["path"]))
    if not matches: raise LookupError(f"no GitLab match for {item['name']} ({item['rows']}×{item['features']})")
    return max(matches)

def _slug(name): return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

def _fetch(url, path):
    if path.exists() and path.stat().st_size: return
    part = path.with_suffix(path.suffix+".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as output:
        while chunk := response.read(1024*1024): output.write(chunk)
    part.replace(path)

def _valid_parquet(path):
    if not path.exists() or not path.stat().st_size: return False
    try: pq.ParquetFile(path); return True
    except Exception: return False

def _download(item, output_dir):
    dataset_id = _resolve(item)
    path = Path(output_dir)/_slug(item["name"])
    path.mkdir(parents=True, exist_ok=True)
    base = f"https://gitlab.com/data/d/openml/{dataset_id}/-/raw/master/dataset"
    for name in ("metadata.json","features.json","qualities.json"):
        _fetch(f"{base}/{name}", path/name)
    _fetch(f"{base}/tables/data.pq", path/"data.pq")
    if not _valid_parquet(path/"data.pq"): raise ValueError("mirror file is not valid Parquet")
    print(f"downloaded {item['name']}", flush=True)
    return dataset_id,path

def _read_manifest(path):
    with open(path, newline="") as handle: return list(csv.DictReader(handle))

def _write_manifest(path, rows):
    fields = list(rows[0])
    if "openml_id" not in fields: fields.append("openml_id")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)

@call_parse
def main(
    manifest:str="meta/amlb/datasets.csv", # Dataset name, dimensions, task, classes, and optional OpenML dataset ID
    output_dir:str=".data/meta_benchmark/amlb", # Download directory
    workers:int=8, # Concurrent downloads
    timeout:int=120, # Maximum seconds per dataset
    limit:int=None, # Optional number of pending datasets
):
    "Download AMLB datasets directly from the GitLab OpenML mirror."
    rows = _read_manifest(manifest)
    pending = [row for row in rows if not _valid_parquet(Path(output_dir)/_slug(row["name"])/"data.pq")]
    if limit is not None: pending = pending[:limit]
    results = asyncio.run(parallel_async(_download, pending, output_dir, n_workers=workers, timeout=timeout, return_exceptions=True))
    complete = failed = 0
    for row,result in zip(pending,results):
        if isinstance(result,Exception):
            failed += 1
            print(f"failed {row['name']}: {type(result).__name__}: {result}")
        else:
            dataset_id,path = result
            row["openml_id"] = dataset_id
            complete += 1
    _write_manifest(manifest, rows)
    print(f"{complete} complete · {failed} failed")
