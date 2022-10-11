import hailtop.batch as hb
import pickle
from tqdm import tqdm

# Just need to load a pickle of triplets from a get_comps function `(comp, c_i, c_j)`
# Could make this into a library, import it, and pass the triplet as a function
# argument. I just prefer to run it on bash
triplet_FN = "customLibs___Isocortexlibs___0509generous"
with open(f"/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/10_Per_Region_CTs/{triplet_FN}___triplet.pickle", "rb") as f:
    thisTriplet = pickle.load(f)

backend = hb.ServiceBackend(billing_project = 'Mouse-Brain-Atlas',
                            # Not used much but hail can deposit things here.
                            # Should clear out periodically or TODO set up with
                            # a lifecycle where just deletes after 20 days
                            remote_tmpdir='gs://macosko_data/jlanglie/hail_tmp')

b = hb.Batch(backend=backend,
             name='atlasdiffexp',
             # See docker image, need to upload to gcr
             default_image="gcr.io/mouse-brain-atlas/jonahatlas:latest",
             # Uses the N1 set of machines. So highmem N1 gives ~7 GB memory per
             # core. If you ask for more memory per core, will just bump up the
             # CPU number (look in hail batch for 'actual' vs 'requested' mem). So to make sure always 2cores highmem, requested 12.5G
             default_memory="12.5Gi",
             # Powers of 2 starting from 1/2. 4 seemed unecessary
             default_cpu=2,
             #  Min 10GB per instance but can ask for more 
             default_storage='9Gi',
             # 18k = 5 hours. Just in case a job hangs or something, can be NULL=inf=24 hour preemtible
             default_timeout=18000, #sec
             # Basically doesn't do anything
             cancel_after_n_failures=3
             )


# Since the comparison triplet-tuple is so small, not worth exporting one
# pickle per job. Just export one and then have hail take a range to run (Can
# just submit one hail job per run, but since startup takes time (docker image
# loading + pickle loading), can be worth it to amortize cost by running a few
# diffexp runs per job with nested parallelism)
# Tuples are exclusive end
def makeRange(start, stopExc, step):
    stopInc = stopExc-1
    return [(n, min(n+step, stopInc)) for n in range(start, stopInc, step)]

allPickleFNs = []

# len pickle input. Could also do start + distance but then need to min with max
for (start, end) in tqdm(makeRange(0, len(thisTriplet)+1, 1)):
    j = b.new_bash_job(name=f'testatlas{start}')

    # Have a set of files need for each job. Let hail take care of downloading
    # from GCS and depositing into random location (then we move into root)
    FN_need = ["atlas_500umis_mt-1pct_cells.txt.gz", "gene_list.txt",
               "clusters.pickle", "cluster_nz_arr.pickle", "cluster_count_arr.pickle",
               "hail_diffexp_runner.py"]

    # Get files and put into root. Can also paramaterize runner function based
    # on the input_file path, but mv doesn't take any time
    for FN in FN_need:
        this_input = b.read_input(f"gs://macosko_data/jlanglie/hail_MBASS_files/{FN}")
        j.command(f'mv {this_input} /{FN}; du -shc /{FN}')

    mainInput = b.read_input(f"gs://macosko_data/jlanglie/hail_MBASS_files/{triplet_FN}___triplet.pickle")

    j.command("chmod -R 777 /io/batch")

    # Optional, I like seeing `top` or some progress bar per job. So this is
    # bash-backgrounded with &, and gets top. See dockerfile
    j.command("bash /getcpu.sh &")

    # Messed around with BLAS/NUMBA environment variables profiling. Overloading
    # 10 threads with 2 cores was faster, with slight diminishing returns but 1.5-2x faster
    j.command(f"su mambauser; . ~/.bashrc ; NUMEXPR_NUM_THREADS=10 OMP_NUM_THREADS=10 "+
              f"OPENBLAS_NUM_THREADS=10 MKL_NUM_THREADS=10 NUMBA_NUM_THREADS=10 "+
              f"python3 -u /hail_diffexp_runner.py --inpickle {mainInput} --outpickle {j.ofile} "+
              f"--daskworkers 6 --daskthreads=2 --start {start} --endexc {end}")

    # Let hail go from j.ofile -> the GCS filename. Assumes triplet_FN is
    # globally unique forever (add UUID)
    thisOutPick = f"gs://macosko_data/jlanglie/hail_out/DiffexpOut_{triplet_FN}___{start}.pickle"
    allPickleFNs.append(thisOutPick)
    b.write_output(j.ofile, thisOutPick)

# Takes all those jobs in the for loop above and spawns into hail batch, pending
# until all finished
b.run()


# ============

# At the end of this deposits into lots of mini pickles on GCS. You can decide
# how to read it in. I don't think you'll like my way, I use the toolz pipe
# style, spawn some gsutil cp jobs, then load it in and merge the output. I
# don't like it, can use a python wrapper around GCS since they're such small files

# I got in the habit of using bash-pipe-esque analysis
from toolz.curried import pipe, get, get_in
from toolz.curried import map as map
from toolz.curried import filter as filter
from toolz.curried import *
import toolz
import pprint
from collections import Counter

import sys
sys.path.append("/home/jlanglie/03_RCTD_Subtypes/96_python_prelude/")
from prelude import *


# Have to clear beforehand, bleh, cleanup
os.system("rm -r /tmp/DE")
with open("/tmp/do.cmd", "w") as f:
    # For use in GNU parallel but so small not worth it
    # pip(allPickleFNs,
    #  map(lambda fn: f"mkdir -p /tmp/DE; gsutil cp {fn} /tmp/DE\n"),
    #  lambda l: f.writelines(l)
    # )
    pip(allPickleFNs,
        map(lambda fn: os.system(f"mkdir -p /tmp/DE; gsutil cp {fn} /tmp/DE\n"))
        )

import pathlib
allFPs = [fp.absolute() for fp in pathlib.Path("/tmp/DE").glob('**/*')]
allDEOuts = []
for thisFP in tqdm(allFPs):
    with open(thisFP, "rb") as f:
        allDEOuts.append(pickle.load(f))

assert pip(allDEOuts,
     map(dictKeys), map(list),
     lambda l: sum(l, []),
     lambda l: len(set(l)) == len(l)
    ), "Duplicate key entries, weird"

# Merge all those docs together.
# Don't know the sum operator for bitwise or. But checked above keys are unique
# so ok to merge like this
allDEOutsDict = pip(allDEOuts,
    lambda l: reduce(lambda a,b: a|b, l)
   )


# Final write to final location of merged pickles
with open(f"/home/jlanglie/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/09_DiffExp/{triplet_FN}.pickle",
          "wb") as f:
    pickle.dump(allDEOutsDict, f)
