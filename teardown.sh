#!/bin/bash
# Tear down the HPNTS AWS validation environment after each run.
# Run this every time -- 2x c6i.xlarge left running unattended adds up fast.
set -euo pipefail
cd "$(dirname "$0")"
terraform destroy -auto-approve
echo "All HPNTS validation instances destroyed."
