import subprocess
from pathlib import Path

speeds = [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 25]
start_positions = [
    20, 30, 40, 50, 60,
    70, 80, 90, 100
    ]

project_dir = Path.home() / "5g-handover-work" / "5g-handover-ml"
ns3_dir = Path.home() / "5g-handover-work" / "ns-3-dev"

output_dir = project_dir / "data" / "raw" / "scenarios"
output_dir.mkdir(parents=True, exist_ok=True)

run_id = 1

for speed in speeds:
    for start_x in start_positions:

        filename = f"speed{speed}_start{start_x}.csv"
        output_file = output_dir / filename

        command = [
            "./ns3",
            "run",
            (
                f"scratch/ml-handover "
                f"--speed={speed} "
                f"--startX={start_x} "
                f"--simTime=60 "
                f"--runId={run_id} "
                f"--output={output_file}"
            ),
        ]

        print(
            f"Running scenario: "
            f"speed={speed} m/s, StartX={start_x} m"
        )

        subprocess.run(
            command,
            cwd=ns3_dir,
            check=True,
        )

        run_id += 1

print ("\nAll scenarios finished.")