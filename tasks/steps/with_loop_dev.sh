#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause
# Run a command with a small loop device attached, its path in
# OPENDS_TEST_FOREIGN_BLKDEV: a whole-disk block device that is not the ublk
# device qublk exports, which raw block mode must refuse. The loop device and
# its backing file go away when the command exits, pass or fail, and the
# command's exit status is kept.
set -e

if [ $# -lt 1 ]; then
	echo "usage: with_loop_dev.sh COMMAND [ARG...]" >&2
	exit 1
fi

img=$(mktemp /tmp/opends_loop.XXXXXX)
trap 'rm -f "$img"' EXIT
truncate -s 1M "$img"
dev=$(losetup --find --show "$img")
trap 'losetup -d "$dev"; rm -f "$img"' EXIT

OPENDS_TEST_FOREIGN_BLKDEV=$dev "$@"
