# LData

A collection of language-based datasets and benchmarks, by Aalto LaReL research group.

## Definitions

This package deals mostly with the data part of the LLM R&D workflow. It contains mainly two abstractions defined as interfaces: a `Dataset` and a `Benchmark`.

### Dataset

A `Dataset` is a collection of data units split into train and test arrays. While it is possible to create an shuffled splits of the data, we provide predefined train/test splits so that you can compare different methods under the same conditions, as well as to ensure reproducibility in this regard. Datasets can be used to pre-train, fine-tune or test models and methods in isolation.

#### Data files

Data files used to create datasets must follow the following format:

- They must be `CSV` files with comma-separated columns and newline-separated rows.
- The first row must be the header `SAMPLE,TARGET`.
- All values in the table must be strings, but they must not be enclosed by any type of delimiter (e.g., single or double quotes).

### Benchmark

A `Benchmark` is a `Dataset` that also provides a score reflecting the performance of the method being evaluated on a given task. Scores are encouraged to be within the range `[0.0, 1.0]` and to increase linearly with the quality of the method's results on the task. Benchmarks are useful to compare multiple models/methods empirically.
