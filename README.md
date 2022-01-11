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
