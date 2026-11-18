# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated local Ray control plane and GPU TorchTrainer workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from grana.config import load_config
from grana.runtime import configure_cuda_allocator_defaults, require_owned_environment


def worker(config):
    from grana.runtime import require_ray_worker
    require_ray_worker()
    import torch
    from ray.train.torch import get_device
    torch.set_num_threads(config["training"]["cpus_per_worker"])
    torch.cuda.set_device(get_device())
    from grana.training.engine import train
    train(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    require_owned_environment()
    allocator_config = configure_cuda_allocator_defaults()
    cfg = load_config(args.config, args.set)
    output = Path(cfg["training"]["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg["training"]["output_dir"] = str(output)
    (output / "resolved_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    # FLA settings come from the venv FLA profile (scripts/fla_profile.py); Ray workers inherit this env.
    os.environ.update({"TOKENIZERS_PARALLELISM": "false", "USE_TF": "0", "TRANSFORMERS_NO_TF": "1",
                       "RAY_TRAIN_V2_ENABLED": "0"})
    # Grana uses the Train V1 session and checkpoint-free Result.metrics APIs.
    import ray
    from ray.train import RunConfig, ScalingConfig, FailureConfig
    from ray.train.torch import TorchConfig, TorchTrainer
    started = time.time()
    # Explicit 'local' never attaches to or stops an existing Ray cluster.
    runtime_dir = tempfile.mkdtemp(prefix="grana-ray-")
    manifest = {"status": "running", "runtime_dir": runtime_dir, "driver_pid": os.getpid(), "output_dir": str(output), "started_unix": started}
    manifest["cuda_allocator_config"] = allocator_config
    manifest.update(driver_ppid=os.getppid(), driver_pgid=os.getpgid(0), driver_sid=os.getsid(0),
                    driver_proc_stat=Path("/proc/self/stat").read_text(),
                    driver_cgroup=Path("/proc/self/cgroup").read_text(),
                    python=sys.executable, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    ray_version=ray.__version__, ray_train_v2_enabled=os.environ["RAY_TRAIN_V2_ENABLED"],
                    lifecycle_owner="foreground canonical Ray launcher; caller must preserve and monitor its session",
                    checkpoint_initialization="weights_only; optimizer, scheduler and RNG start fresh")
    source_files = sorted(list((ROOT / "grana").rglob("*.py")) + list((ROOT / "scripts/ray_migration").glob("*.py")))
    manifest["code_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    manifest_path = output / "ray_run.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        ray.init(address="local", _node_ip_address="127.0.0.1", dashboard_host="127.0.0.1",
                 _temp_dir=runtime_dir, include_dashboard=False,
                 num_cpus=max(2, cfg["training"]["num_workers"] * cfg["training"]["cpus_per_worker"] + 1),
                 log_to_driver=True)
        trainer = TorchTrainer(
            worker, train_loop_config=cfg,
            torch_config=TorchConfig(timeout_s=cfg["training"]["distributed_timeout_seconds"]),
            scaling_config=ScalingConfig(num_workers=cfg["training"]["num_workers"], use_gpu=True,
                                         resources_per_worker={"CPU": cfg["training"]["cpus_per_worker"], "GPU": 1}),
            run_config=RunConfig(name="train", storage_path=str(output / "ray"), failure_config=FailureConfig(max_failures=0)),
        )
        result = trainer.fit()
        if result.error:
            raise RuntimeError(str(result.error))
        manifest.update(status="succeeded", metrics=result.metrics)
        receipt_path = output / "wandb_run.json"
        if receipt_path.is_file():
            manifest["logging"] = json.loads(receipt_path.read_text())
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["elapsed_seconds"] = time.time() - started
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        ray.shutdown()


if __name__ == "__main__":
    main()
