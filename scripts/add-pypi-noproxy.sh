#!/usr/bin/env bash
#
# Add PyPI domains to the docker daemon NO_PROXY so build containers
# bypass the proxy for PyPI.
#
# Usage (as root):
#   sudo bash scripts/add-pypi-noproxy.sh
#
# An optional first argument overrides the config path (used for testing).
set -euo pipefail

CONF=${1:-/etc/systemd/system/docker.service.d/http-proxy.conf}
DOMAINS=(pypi.tuna.tsinghua.edu.cn pypi.org)

if [[ ! -f $CONF ]]; then
    echo "error: $CONF not found" >&2
    exit 1
fi

if ! grep -q '^Environment="NO_PROXY=' "$CONF"; then
    echo "error: no NO_PROXY line in $CONF" >&2
    exit 1
fi

timestamp=$(date +%Y%m%d%H%M%S)
cp -a "$CONF" "$CONF.bak.$timestamp"
echo "backup: $CONF.bak.$timestamp"

changed=0
for domain in "${DOMAINS[@]}"; do
    if ! grep -q "$domain" "$CONF"; then
        sed -i "s/^\(Environment=\"NO_PROXY=[^\"]*\)\"/\1,$domain\"/" "$CONF"
        echo "added $domain to NO_PROXY"
        changed=1
    fi
done

if [[ $changed -eq 0 ]]; then
    echo "NO_PROXY already contains all PyPI domains; no change"
fi

grep '^Environment="NO_PROXY=' "$CONF"

echo
read -r -p "Reload systemd and restart docker (restarts running containers)? [y/N] " answer
case "$answer" in
    [yY] | [yY][eE][sS])
        systemctl daemon-reload
        systemctl restart docker
        echo "docker restarted; PyPI traffic in builds now bypasses the proxy"
        ;;
    *)
        echo "apply manually: sudo systemctl daemon-reload && sudo systemctl restart docker"
        ;;
esac
