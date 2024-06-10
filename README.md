# LData

A collection of language-based datasets and benchmarks.
This package deals mostly with the data part of the LLM R&D workflow. It contains mainly two abstractions: a `Dataset` and a `Benchmark`.

## Dataset

A `Dataset` is a collection of data units split into train and test arrays. While it is possible to create shuffled splits of the data, we provide tools to ensure reproducibility, so that you can compare different methods/models under the same conditions. Datasets can be used to pre-train, fine-tune or test models and methods in isolation.

### Data files

Data files used to create datasets must follow the following format:

- They must be `CSV` files with comma-separated columns and newline-separated rows.
- The first row must be the header `SAMPLE,TARGET`.
- All values in the table must be strings, but they must not be enclosed by any type of delimiter (e.g., single or double quotes).

## Benchmark

A `Benchmark` is a `Dataset` that also provides a score reflecting the performance of the method being evaluated on a given task. Scores are encouraged to be within the range `[0.0, 1.0]` and to increase linearly with the quality of the method's results on the task. Benchmarks are useful to compare multiple models/methods empirically.

## Creating new datasets and benchmarks

This package is supposed to be a growing library of datasets and benchmarks for everyone to use; the more the better! You are encouraged to contribute new datasets or wrappers of existing ones. To be merged into this repository, datasets or benchmarks must faithfully implement the `Dataset` and `Benchmark` abstract classes, respectively. If you think there is something wrong or missing with these classes, feel free to open an issue!

## Downloading datasets and benchmarks

Currently, when you install this package, you will also download all datasets and benchmarks available in the `data` folder. This is not an issue at the moment, but as we scale up the number of datasets, we have in our "to-do" list to provide a sub-package for each available dataset, as well as for groups of them, so that you can install in your system the core components of the library plus only the datasets you need. Coming soon (hopefully)!.
