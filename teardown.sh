
set -euo pipefail
cd "$(dirname "$0")"
terraform destroy -auto-approve
echo "All HPNTS validation instances destroyed."
