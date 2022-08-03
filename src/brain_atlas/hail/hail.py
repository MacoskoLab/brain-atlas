import click
import hailtop.batch as hb


@click.command("hail")
@click.option("--batch-name", default="atlas-de")
@click.option("--triplet-file", required=True, help="Path to DE triplet pickle on GCS")
@click.option("--cluster-data", required=True, help="Path to cluster npz on GCS")
@click.option("--n-jobs", type=int, required=True, help="Number of jobs to run")
@click.option("--step", type=int, default=1)
@click.option("--project", default="Mouse-Brain-Atlas")
@click.option("--gs-output", default="gs://macosko_data/jwebber/hail_output")
@click.option("--remote-tmp", default="gs://macosko_data/jwebber/hail_tmp")
@click.option("--default-image", default="gcr.io/mouse-brain-atlas/hail-de:latest")
def main(
    batch_name: str,
    triplet_file: str,
    cluster_data: str,
    n_jobs: int,
    step: int,
    project: str,
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

        # Get data file and put into root. Can also paramaterize runner function based
        # on the input_file path, but mv doesn't take any time
        local_data_path = b.read_input(cluster_data)
        main_input = b.read_input(triplet_file)

        # maybe only necessary because it was using the 'mamba user'?
        j.command("chmod -R 777 /io/batch")

        # Messed around with BLAS/NUMBA environment variables profiling. Overloading
        # 10 threads with 2 cores was faster, with slight diminishing returns but 1.5-2x faster
        j.command(
            " ".join(
                (
                    "NUMEXPR_NUM_THREADS=10",
                    "OMP_NUM_THREADS=10",
                    "OPENBLAS_NUM_THREADS=10",
                    "MKL_NUM_THREADS=10",
                    "NUMBA_NUM_THREADS=10",
                    "hail-de",
                    "--input-file",
                    f"{main_input}",
                    "--output-file",
                    f"{j.ofile}",
                    "--cluster-data",
                    f"{local_data_path}",
                    "--start",
                    f"{start}",
                    "--endexc",
                    f"{end}",
                )
            )
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
