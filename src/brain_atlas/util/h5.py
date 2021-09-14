
import scipy.sparse
import tables


def read_10x_h5(path):
    with tables.open_file(str(path), "r") as f:
        dsets = {}
        for node in f.walk_nodes('/matrix', 'Array'):
            dsets[node.name] = node.read()

        M, N = dsets['shape']
        data = dsets['data']

        matrix = scipy.sparse.csr_matrix(
            (data, dsets['indices'], dsets['indptr']),
            shape=(N, M),
        )

        return (
            matrix,
            dsets['barcodes'].astype(str),
            tuple(dsets["name"].astype(str)),
            tuple(dsets["id"].astype(str))
        )


def read_10x_h5_meta(path):
    with tables.open_file(str(path), "r") as f:
        dsets = {}
        for node in f.walk_nodes('/matrix', 'Array'):
            if node.name in {"name", "id"}:
                dsets[node.name] = node.read()

        return (
            tuple(dsets["name"].astype(str)),
            tuple(dsets["id"].astype(str))
        )

