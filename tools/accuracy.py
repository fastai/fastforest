"CLI for `fastforest.bench.compare`: accuracy and timing on canonical dataset splits."
from fastcore.script import call_parse

from fastforest.bench import compare

main = call_parse(compare)
