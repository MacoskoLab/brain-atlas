The goal of this is to use Hail Batch to parallelize diffexp runs.

The main idea is that the only state a run needs is the constant atlas files (see below) + and the 'diffexp triplet' = the result of `get_comps` = `(comp, c_i, c_j)`.

There are main files: 

- hail_diffexp_runner.py

The actual script run on hail docker instances. This takes as input {the
constant atlas files, the list of diffexp triplets, and the diffexp triplet
indices to actuall run} and then runs `find_genes.generic_de` on those
triplets. It returns a pickle file of the result

- hail_diffexp_coach.py

The script which interfaces with hail batch (the coach manages the runner :-) ). It sets the instance parameters (cpu, memory, docker image link) and spawns one job per diffexp triplet. It also collates the result in a hacky way

- Dockerfile

This installes the conda environment as needed for brain-atlas

# Getting ready to run

- TODO some hardcoded GCS paths. Worth changing/parameterizing so no collision, but I can change my paths
- Clone the brain-atlas github in the Dockerfile folder (at `00_CLONE_BRAIN_ATLAS_HERE`), so it's included in the dockerimage
- Upload the constant atlas files to a bucket. See 13_precalc_for_hail. 13_precalc_for_hail gives the location in the code where I made them, and the GCS path you'll need to change
- If change the docker, make a new gcr docker file
