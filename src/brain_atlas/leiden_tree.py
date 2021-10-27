from pathlib import Path
from typing import Optional, Sequence

import dask.array as da
import yaml


class LeidenTree:
    """Object that represents a hierarchical tree of leiden clustering.

    All files will be stored under the root path, except for the raw counts,
    which can be stored elsewhere. This object can represent a subtree.

    :param tree_dir: Directory containing this (sub)tree
    :param data: Location of the raw data (should be a zarr array)
    :param n_pcs: number of PCs to use, if computing PCA
    :param k_neighbors: number of neighbors for kNN
    :param transform: transform for scaling counts (sqrt, log1p, or none)
    :param resolution: the Leiden resolution used to define the clusters
    """

    def __init__(
        self,
        tree_dir: Path,
        data: Path,
        n_pcs: Optional[int],
        k_neighbors: int,
        transform: str = None,
        resolution: str = None,
    ):
        self.dir = tree_dir
        self.dir.mkdir(exist_ok=True)

        assert data.exists(), f"{data} does not exist"
        self.data = data

        assert n_pcs is None or n_pcs > 0
        self.n_pcs = n_pcs

        assert k_neighbors > 0
        self.k_neighbors = k_neighbors

        if transform is None or transform.lower() == "none":
            self.transform = None
        elif transform.lower() in ("sqrt", "log1p"):
            self.transform = transform.lower()
        else:
            raise ValueError(f"Unknown transform {transform}")

        self.resolution = resolution

    @staticmethod
    def read_yaml(yaml_path: Path):
        with yaml_path.open() as fh:
            metadata = yaml.safe_load(fh)

        return {**metadata, "data": Path(metadata["data"])}

    @staticmethod
    def from_path(tree_path: Path):
        metadata_path = tree_path / "metadata.yaml"
        assert metadata_path.exists(), f"{metadata_path} does not exist"

        metadata = LeidenTree.read_yaml(metadata_path)

        return LeidenTree(tree_dir=tree_path, **metadata)

    def write_metadata(self):
        with self.metadata_yaml.open("w") as out:
            # convert Path to string before writing
            yaml.safe_dump({**self.metadata, "data": str(self.data)}, stream=out)

    def is_valid_cache(self):
        if not self.metadata_yaml.exists():
            return False

        metadata = LeidenTree.read_yaml(self.metadata_yaml)

        return self.metadata == metadata

    def subcluster_path(self, level: Sequence[int]):
        return self.dir.joinpath(*map(str, level))

    @property
    def transform_fn(self):
        if self.transform is None:
            return lambda x: x
        elif self.transform == "sqrt":
            return da.sqrt
        elif self.transform == "log1p":
            return da.log1p
        else:
            raise ValueError(f"Unknown transform {self.transform}")

    @property
    def metadata(self):
        return {
            "data": self.data,
            "n_pcs": self.n_pcs,
            "k_neighbors": self.k_neighbors,
            "transform": self.transform,
            "resolution": self.resolution,
        }

    @property
    def metadata_yaml(self):
        return self.dir / "metadata.yaml"

    @property
    def selected_genes(self):
        return self.dir / "selected_genes.npz"

    @property
    def pca(self):
        return self.dir / "pca.zarr"

    @property
    def knn(self):
        return self.dir / "knn.zarr"

    @property
    def snn(self):
        return self.dir / "snn.zarr"

    @property
    def clustering(self):
        return self.dir / "clusters.npz"

    def __str__(self):
        return str(self.dir)
