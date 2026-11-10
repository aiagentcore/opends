#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Run a command with two block devices that raw block mode must refuse, their
# paths in the environment:
#   OPENDS_TEST_FOREIGN_BLKDEV  a small loop device: a whole disk, but not a
#                               ublk disk;
#   OPENDS_TEST_FOREIGN_UBLK    a ublk disk of the same size as UBLK_DEV that
#                               a second qublk serves from a sparse file: a
#                               ublk disk, but not the namespace. qublk
#                               serves no NVMe ioctl device for a file, so
#                               the driver refuses it for that; a disk that
#                               answers with another serial number or
#                               namespace id needs a second NVMe and is not
#                               covered.
# Both devices and their backing files go away when the command exits, pass or
# fail, and the command's exit status is kept.
set -e

if [ $# -lt 2 ]; then
	echo "usage: with_foreign_blkdevs.sh UBLK_DEV COMMAND [ARG...]" >&2
	exit 1
fi

ublk=$1
shift

dir=$(mktemp -d /tmp/opends_foreign.XXXXXX)
loop=
qpid=
cleanup() {
	if [ -n "$qpid" ]; then
		kill "$qpid" 2>/dev/null || true
		wait "$qpid" 2>/dev/null || true
	fi
	if [ -n "$loop" ]; then
		losetup -d "$loop"
	fi
	rm -rf "$dir"
}
trap cleanup EXIT

truncate -s 1M "$dir/loop.img"
loop=$(losetup --find --show "$dir/loop.img")

truncate -s "$(blockdev --getsize64 "$ublk")" "$dir/ublk.img"
qublk serve "$dir/ublk.img" --be linux --nqueues 1 \
	< /dev/null > "$dir/qublk.log" 2>&1 &
qpid=$!
foreign=
for _ in $(seq 1 100); do
	foreign=$(grep -oE '/dev/ublkb[0-9]+ ready' "$dir/qublk.log" | cut -d' ' -f1)
	[ -n "$foreign" ] && break
	sleep 0.1
done
if [ -z "$foreign" ]; then
	echo "error: the second qublk did not expose a ublk device" >&2
	cat "$dir/qublk.log" >&2
	exit 1
fi

OPENDS_TEST_FOREIGN_BLKDEV=$loop OPENDS_TEST_FOREIGN_UBLK=$foreign "$@"
