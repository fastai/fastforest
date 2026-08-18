from pathlib import Path

from fastforest.tools import CLASSIFICATION,REGRESSION,DatasetView,load_datasets,load_results,mk_table

root = Path(__file__).parents[1]
results = root/"tools"/"results"

def render_readme():
    "Generate README.md from its template and benchmark result CSVs."
    datasets = load_datasets(results/"datasets.csv")
    regression = load_results(results/"regression.csv", datasets)
    classification = load_results(results/"classification.csv", datasets)
    tables = {
        "SUMMARY_REGRESSION": mk_table(regression, [DatasetView("sgemm"), DatasetView("rossmann")], REGRESSION, "summary"),
        "SUMMARY_CLASSIFICATION": mk_table(classification,
            [DatasetView("covertype_bin"), DatasetView("adult")], CLASSIFICATION, "summary"),
        "BENCHMARK_REGRESSION": mk_table(regression,
            [DatasetView(name) for name in ("california", "concrete", "diamonds", "allstate", "diabetes", "bluebook", "walmart", "ashrae")], REGRESSION),
        "BENCHMARK_CLASSIFICATION": mk_table(classification,
            [DatasetView(name) for name in ("bank", "click", "shuttle", "airlines", "higgs", "sf_police", "kddcup99")], CLASSIFICATION),
        "GROUPED_COVERTYPE": mk_table(classification,
            [DatasetView("covertype_group", comparison="covertype_bin")], CLASSIFICATION),
    }
    readme = (root/"README.tmpl").read_text()
    for name,table in tables.items():
        token = "{{"+name+"}}"
        if readme.count(token) != 1: raise ValueError(f"README.tmpl must contain exactly one {token}")
        readme = readme.replace(token, table)
    (root/"README.md").write_text(readme)

if __name__ == "__main__": render_readme()
