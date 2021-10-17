from pathlib import Path
from typing import Sequence

import yaml


class LeidenTree:
    """Object that represents a hierarchical tree of leiden clustering.

    All files will be stored under the root path, except for the raw counts,
    which can be stored elsewhere. This object can represent a subtree.
    """

    def __init__(
        self,
        root: Path,
        data: Path,
        n_pcs: int,
        k_neighbors: int,
        resolution: str = None,
    ):
        self.root = root
        self.root.mkdir(exist_ok=True)

        self.data = data
        assert self.data.exists(), f"{data} does not exist"

        assert n_pcs > 0
        self.n_pcs = n_pcs

        assert k_neighbors > 0
        self.k_neighbors = k_neighbors

        self.resolution = resolution

    @staticmethod
    def from_path(root_path: Path):
        metadata_path = root_path / "metadata.yaml"
        assert metadata_path.exists(), f"{metadata_path} does not exist"

        with metadata_path.open() as fh:
            metadata = yaml.safe_load(fh)

        return LeidenTree(
            root=root_path,
            data=Path(metadata["data"]),
            n_pcs=metadata["n_pcs"],
            k_neighbors=metadata["k_neighbors"],
            resolution=metadata["resolution"],
        )

    def write_metadata(self):
        metadata = {
            "data": self.data,
            "n_pcs": self.n_pcs,
            "k_neighbors": self.k_neighbors,
            "resolution": self.resolution,
        }

        with self.metadata.open("w") as out:
            yaml.safe_dump(metadata, stream=out)

    def is_valid_cache(self):
        if not self.metadata.exists():
            return False

        with self.metadata.open() as fh:
            metadata = yaml.safe_load(fh)

        return (
            self.data == Path(metadata["data"])
            and self.n_pcs == metadata["n_pcs"]
            and self.k_neighbors == metadata["k_neighbors"]
            and self.resolution == metadata["resolution"]
        )

    @property
    def metadata(self):
        return self.root / "metadata.yaml"

    @property
    def selected_genes(self):
        return self.root / "selected_genes.npz"

    @property
    def pca(self):
        return self.root / "pca.zarr"

    @property
    def knn(self):
        return self.root / "knn.zarr"

    @property
    def snn(self):
        return self.root / "snn.zarr"

    @property
    def clustering(self):
        return self.root / "clusters.npz"

    def subcluster_path(self, level: Sequence[int]):
        return self.root.joinpath(*map(str, level))

    def __str__(self):
        return str(self.root)
