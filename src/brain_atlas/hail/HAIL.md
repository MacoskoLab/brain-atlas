The goal of this is to use Hail Batch to parallelize diffexp runs. It's worth noting that in some cases it will be more efficient to use a large machine, which can load all the relevant data into memory.

The main idea is that the only state a run needs is the constant atlas files (see below) + and the 'diffexp triplet' = the result of `get_comps` = `(comp, c_i, c_j)`.

There are two main files:

`hail.py` - When the package is installed locally, this is available as the `hail` command in the terminal. This launches a bunch of DE jobs via Hail. Right now it assumes that you want to do all-by-all but that's not actually feasible.

It requires certain files to be uploaded to GCS:
 - the `node-tree` is a pickle of the node tree (from `util.tree.to_tree`)
 - the `cluster-data` file is an `npz` containing three arrays: the cluster ids (as ints), a nonzero count array, and the cluster count array.

`client.py` - The actual script runs on hail docker instances. This takes the above files as input along with a slice into the list of comparisons. It returns a pickle file of the result.

A better way to run this would be if the client had some different options for what kind of comparisons to do.

`Dockerfile_hail` contains the dependencies for the client script&emdash;to save space, it does not have everything needed for the whole package. It has been built and pushed to `gcr.io/mouse-brain-atlas/hail-de`.
