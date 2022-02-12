## Mouse Brain Atlas analysis

This repository contains a scripts and tools for clustering sc/snRNAseq data. Specifically, it has been used to cluster roughly 6M single-nuclei profiles generated in the Macosko lab.

The sequenced libraries were aligned with [cellranger count](https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/using/count) and then processed with [CellBender remove-background](https://cellbender.readthedocs.io/en/latest/usage/index.html).

### Outline

We make heavy use of [Dask](https://dask.org/) and [Zarr](https://zarr.readthedocs.io/). This combination enables the largest computations to be performed in parallel chunks without ever loading the full dataset into memory. Analysis was performed on a very large (`n2d-highmem-64`) virtual machine on Google Cloud.

The overall procedure (listed values are parameters that can be changed):

 1. Aggregate all processed h5 files into one Zarr array, which is stored on disk
 2. Filter to samples with ≥500 UMIs and <1% UMIs mapping to mitochondrial genes
 3. Iterative clustering, starting with the full array:
    1. Gene selection via a binomial approximation of the sequencing process. Genes that are >5% below expected frequency are selected
    2. Square-root transformation of the raw counts
       1. log1p is also possible, as well as scaling and PCA (using [Dask ML](https://ml.dask.org/modules/generated/dask_ml.decomposition.IncrementalPCA.html)), but these steps were not used for our analysis
    3. kNN construction using [pynndescent](https://pynndescent.readthedocs.io/), with cosine distance
    4. SNN computed from kNN, implemented to run in parallel with [numba](https://numba.pydata.org/)
    5. Leiden clustering over a range of resolutions, using the [leidenalg](https://leidenalg.readthedocs.io/en/latest/intro.html) package. We choose the first resolution that yields a non-trivial clustering (*i.e.* more than one cluster).
    6. Subcluster the results until either a) no further clusters are found or more commonly b) the resulting clusters have no DE genes between them.

The clustering is stored in a nested directory structure, with a helper class defined in the `leiden_tree` module.

## Installation

The best way to install this package is in a new conda repository. The `environment.yaml` file makes this simple:

```shell
git clone https://github.com/MacoskoLab/brain-atlas.git  # clone this repository
cd brain-atlas
conda env create -f environment.yaml  # creates an environment named `atlas`
conda activate atlas
```

You can specify an alternative name for the environment with the option `-n [ENV_NAME]`.

If you're installing the package on a GCP instance, your credentials should already be configured. Otherwise you may need to authenticate:

```shell
gcloud init
gcloud auth login
gcloud auth application-default login  # not sure if this is needed

# ... follow instructions
```

## Querying the data

The `atlas query` command will subsample a subtree from the full clustering and save the count data and metadata as a compressed numpy array (`.npz`) and `csv` file respectively. The interface looks like this:

```
➜ atlas query
2022-01-20 15:44:17 - brain_atlas.scripts.__init__ - INFO - No Dask cluster found
2022-01-20 15:44:17 - brain_atlas.scripts.__init__ - INFO - Starting cluster on local machine
Usage: atlas query [OPTIONS] [QUERY]...

  Extracts QUERY from a cluster array, subsampling the clusters if needed, and
  creates an npy file containing the counts along with a csv of metadata.

  Can be run on a local zarr directory, or download data from GCS

Options:
  --data-path TEXT        Path to count array  [required]
  --cluster-path TEXT     Path to cluster array  [required]
  --metadata-path TEXT    Path to cell metadata  [required]
  --output-dir DIRECTORY  Directory for output  [required]
  --subsample INTEGER     Subsample the data per cluster  [default: 5000]
  --max-cells INTEGER     Max number of cells to return  [default: 50000]
  --help                  Show this message and exit.
```

`data-path` and `cluster-path` are the paths to [Zarr](https://zarr.readthedocs.io/) arrays containing the counts and cluster labels respectively. These can be local directories or locations on Google Cloud Storage. `metadata-path` is similar, but should point to a [Parquet](https://parquet.apache.org/) dataframe with metadata for the cells.

All three of these arrays will be subset to descendents of the requested `QUERY`, which is a series of integers that defines a subset of the cluster tree, e.g. `0 1 0 2` to select everything beyond that node in the tree. The results are written to `output-dir`.

`subsample` will subsample each leaf cluster to at most the specified number of cells. If the result would be more than `max-cells` it will not be downloaded&mdash;this is a precaution to avoid accidentally downloading very large arrays. To download it anyway, simply increase the limit.
