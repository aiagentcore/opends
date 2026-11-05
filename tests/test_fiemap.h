/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_fiemap.h - FIEMAP a file into its extents, for the raw block mode
 * test and bench: each extent's place in the file and on the block device
 * the file system is mounted on, which in the test suite is the ublk device
 * qublk exports, so the device offset is also the namespace byte offset.
 * tasks/steps/first_extent.py applies the same "plain" rule in Python.
 */
#ifndef TEST_FIEMAP_H_
#define TEST_FIEMAP_H_

#include <errno.h>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/ioctl.h>

/* Data at an extent's device offset is the file's bytes only when none of
 * these is set: no location yet, not written, packed, encrypted or stored
 * with other data. */
#define TEST_FIEMAP_NOT_PLAIN                                                  \
	(FIEMAP_EXTENT_UNKNOWN | FIEMAP_EXTENT_DELALLOC |                      \
	 FIEMAP_EXTENT_ENCODED | FIEMAP_EXTENT_DATA_ENCRYPTED |                \
	 FIEMAP_EXTENT_NOT_ALIGNED | FIEMAP_EXTENT_DATA_INLINE |               \
	 FIEMAP_EXTENT_DATA_TAIL | FIEMAP_EXTENT_UNWRITTEN)

struct test_extent {
	uint64_t file_off;
	uint64_t dev_off;
	uint64_t len;
	bool plain;
};

/* Map the extents of fd into out[0..max). Returns how many were mapped, or
 * -errno. *complete says whether the last of them ends the file, i.e. max
 * was large enough. */
static inline int
test_fiemap(int fd, struct test_extent *out, uint32_t max, bool *complete)
{
	size_t sz = sizeof(struct fiemap) + max * sizeof(struct fiemap_extent);
	struct fiemap *fm = calloc(1, sz);
	if (!fm)
		return -ENOMEM;
	fm->fm_start = 0;
	fm->fm_length = FIEMAP_MAX_OFFSET;
	fm->fm_flags = FIEMAP_FLAG_SYNC;
	fm->fm_extent_count = max;
	if (ioctl(fd, FS_IOC_FIEMAP, fm) < 0) {
		int rc = -errno;
		free(fm);
		return rc;
	}
	uint32_t n = fm->fm_mapped_extents;
	for (uint32_t i = 0; i < n; i++) {
		const struct fiemap_extent *fe = &fm->fm_extents[i];
		out[i].file_off = fe->fe_logical;
		out[i].dev_off = fe->fe_physical;
		out[i].len = fe->fe_length;
		out[i].plain = !(fe->fe_flags & TEST_FIEMAP_NOT_PLAIN);
	}
	*complete =
	        n > 0 && (fm->fm_extents[n - 1].fe_flags & FIEMAP_EXTENT_LAST);
	free(fm);
	return (int)n;
}

#endif /* TEST_FIEMAP_H_ */
