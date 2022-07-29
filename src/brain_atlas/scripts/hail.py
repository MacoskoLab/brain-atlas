import click
import hailtop.batch as hb


@click.command("hail")
@click.option("--batch-name", default="atlas-de")
@click.option("--triplet-file", required=True, help="Name of DE triple pickle on GCS")
@click.option("--n-jobs", type=int, required=True, help="Number of jobs to run")
@click.option("--step", type=int, default=1)
@click.option("--project", default="Mouse-Brain-Atlas")
@click.option("--gs-data", default="gs://macosko_data/jwebber/hail_input")
@click.option("--gs-output", default="gs://macosko_data/jwebber/hail_output")
@click.option("--remote-tmp", default="gs://macosko_data/jwebber/hail_tmp")
@click.option("--default-image", default="gcr.io/mouse-brain-atlas/jonahatlas:latest")
def main(
    batch_name: str,
    triplet_file: str,
    n_jobs: int,
    step: int,
    project: str,
    gs_data: str,
    gs_output: str,
    remote_tmp: str,
    default_image: str,
):
    backend = hb.ServiceBackend(
        billing_project=project,
        # Not used much but hail can deposit things here.
        # Should clear out periodically or set up with
        # a lifecycle where just deletes after 20 days
        remote_tmpdir=remote_tmp,
    )

    b = hb.Batch(
        backend=backend,
        name=batch_name,
        # See docker image, need to upload to gcr
        default_image=default_image,
        # Uses the N1 set of machines. So highmem N1 gives ~7 GB memory per
        # core. If you ask for more memory per core, will just bump up the
        # CPU number (look in hail batch for 'actual' vs 'requested' mem).
        # So to make sure always 2cores highmem, requested 12.5G
        default_memory="12.5Gi",
        # Powers of 2 starting from 1/2. 4 seemed unecessary
        default_cpu=2,
        #  Min 10GB per instance but can ask for more
        default_storage="9Gi",
        # 18k = 5 hours. Just in case a job hangs or something, can be NULL=inf=24 hour preemtible
        default_timeout=18000,  # sec
        # Basically doesn't do anything
        cancel_after_n_failures=3,
    )

    # Since the comparison triplet-tuple is so small, not worth exporting one
    # pickle per job.
    for start in range(0, n_jobs, step):
        end = start + step
        j = b.new_bash_job(name=f"atlas-de-{start}")

        # Have a set of files need for each job. Let hail take care of downloading
        # from GCS and depositing into random location (then we move into root)
        fn_need = [
            "atlas_500umis_mt-1pct_cells.txt.gz",
            "gene_list.txt",
            "clusters.pickle",
            "cluster_nz_arr.pickle",
            "cluster_count_arr.pickle",
            "hail_de_runner.py",
        ]

        # Get files and put into root. Can also paramaterize runner function based
        # on the input_file path, but mv doesn't take any time
        for fn in fn_need:
            this_input = b.read_input(f"{gs_data}/{fn}")
            j.command(f"mv {this_input} /{fn}; du -shc /{fn}")

        main_input = b.read_input(f"{gs_data}/{triplet_file}")

        # maybe only necessary because it was using the 'mamba user'?
        j.command("chmod -R 777 /io/batch")

        # Messed around with BLAS/NUMBA environment variables profiling. Overloading
        # 10 threads with 2 cores was faster, with slight diminishing returns but 1.5-2x faster
        j.command(
            "NUMEXPR_NUM_THREADS=10 OMP_NUM_THREADS=10 "
            "OPENBLAS_NUM_THREADS=10 MKL_NUM_THREADS=10 NUMBA_NUM_THREADS=10 "
            f"hail_de --input-file {main_input} --output-file {j.ofile} "
            f"--start {start} --endexc {end}"
        )

        # Let hail go from j.ofile -> the GCS filename. Assumes triplet_FN is
        # globally unique forever (add UUID)
        this_output = f"{gs_output}/{batch_name}_{start}.pickle"

        b.write_output(j.ofile, this_output)

    # Takes all those jobs in the loop above and spawns into hail batch, pending
    # until all finished
    b.run()


if __name__ == "__main__":
    main()
