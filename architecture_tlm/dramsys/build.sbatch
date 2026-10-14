#!/bin/bash
#SBATCH --job-name=pbit-dram-build
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=build-%j.log
set -euo pipefail
root=${SLURM_SUBMIT_DIR}
flags=()
if [[ -d "$root/vendor/deps" ]]; then
 for name in nlohmann_json fmt dramutils drampower systemclanguage sqlite3; do
  src=$(find "$root/vendor/deps/$name" -mindepth 1 -maxdepth 1 -type d | head -1)
  flags+=("-DFETCHCONTENT_SOURCE_DIR_${name^^}=$src")
 done
fi
cmake -S "$root/architecture_tlm/dramsys" -B "$root/build" \
 -DCMAKE_BUILD_TYPE=Release -DDRAMSYS_SOURCE="$root/vendor/DRAMSys" "${flags[@]}"
cmake --build "$root/build" -j "${SLURM_CPUS_PER_TASK:-4}"
