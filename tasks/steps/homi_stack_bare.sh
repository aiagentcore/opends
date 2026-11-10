#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Reduce a running HOMI stack to its bare form: stop xal-server and unmount
# the file system, leaving homi and qublk up with the ublk device exported and
# nothing on it. Raw block mode needs no more than that; the bare-device test
# runs against this state. homi_stack_down.sh tears the rest down afterwards.
set -e

if [ $# -ne 1 ]; then
	echo "usage: homi_stack_bare.sh MOUNT" >&2
	exit 1
fi

MOUNT=$1

pkill -TERM -x xal-server 2>/dev/null || true
for _ in $(seq 1 120); do
	pgrep -x xal-server > /dev/null || break
	sleep 1
done
if pgrep -x xal-server > /dev/null; then
	echo "error: xal-server did not exit" >&2
	exit 1
fi
rm -f /dev/shm/xal_dev*

if mountpoint -q "$MOUNT"; then
	umount "$MOUNT"
fi

pgrep -x homi > /dev/null || { echo "error: homi is not running" >&2; exit 1; }
pgrep -x qublk > /dev/null || { echo "error: qublk is not running" >&2; exit 1; }
echo "bare stack: homi + qublk ($(cat /run/homi/ublk_dev)), nothing mounted, no index"
