import submitit

def check_env():
    import sys, os, subprocess
    print("=== INSIDE SUBMITIT JOB ===")
    print("PYTHON VERSION:", sys.version)
    print("PYTHON EXECUTABLE:", sys.executable)
    print("which python:", subprocess.check_output("which python", shell=True).decode().strip())
    print("CONDA_DEFAULT_ENV:", os.environ.get("CONDA_DEFAULT_ENV"))
    print("PATH:", os.environ.get("PATH"))
    print("PYTHONPATH:", os.environ.get("PYTHONPATH"))

if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="submitit_logs")

    executor.update_parameters(
        timeout_min=5,
        slurm_partition="a100x4q",
        slurm_nodes=1,
        slurm_ntasks_per_node=1,
        slurm_cpus_per_task=1,     # <-- only 1 CPU now
        slurm_account="yzmkp1",
        # If needed later:
        # slurm_setup=["source ~/.bashrc", "conda activate jepa"],
    )

    job = executor.submit(check_env)
    print("SUBMITTED:", job.job_id)
    job.result()
