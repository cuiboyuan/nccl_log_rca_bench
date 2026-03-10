from pathlib import Path
import json

root = Path("/workspace/nccl_log_project/phase2_runs")

runs = sorted([p for p in root.iterdir() if p.is_dir()])

print(f"Found {len(runs)} phase2 runs\n")

for run in runs:
    meta = run / "metadata.json"
    stdout = run / "stdout.log"
    stderr = run / "stderr.log"
    nccl_logs = list(run.glob("nccl_logs_*.txt"))

    print(f"Run: {run.name}")
    print(f"  metadata: {'yes' if meta.exists() else 'no'}")
    print(f"  stdout:   {'yes' if stdout.exists() else 'no'}")
    print(f"  stderr:   {'yes' if stderr.exists() else 'no'}")
    print(f"  nccl log count: {len(nccl_logs)}")

    if meta.exists():
        with open(meta) as f:
            m = json.load(f)
        print(f"  scenario: {m.get('scenario')}")
        print(f"  fault_type: {m.get('fault_type')}")
        print(f"  affected_rank: {m.get('affected_rank')}")
        print(f"  status: {m.get('status')}")
    print()
