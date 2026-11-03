/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_raw_block.h - Helpers shared by the raw block mode tests: a raw
 * read or write through a registered block device handle, staged through a
 * GPU buffer, and a pattern check.
 */
#ifndef TEST_RAW_BLOCK_H_
#define TEST_RAW_BLOCK_H_

#include "opends.h"
#include "read_pattern.h"
#include "test_cuda_common.h"

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#define ENV_RAW_BLOCK "OPENDS_AISIO_RAW_BLOCK"

/* Compare n bytes of host memory with the pattern starting at offset off.
 * Returns the number of mismatching bytes. */
static inline size_t
check_pattern(const unsigned char *host, uint64_t off, size_t n)
{
	size_t bad = 0;
	for (size_t i = 0; i < n; i++)
		if (host[i] != pattern_byte((off_t)(off + i)))
			bad++;
	return bad;
}

/* Raw read of n bytes at device offset dev_off into gpu, copied to host.
 * Returns the backend's result (bytes or -error). */
static inline ssize_t
raw_read(opends_handle_t fh, void *gpu, unsigned char *host, size_t n,
         uint64_t dev_off)
{
	cuda_buf_zero(gpu, n);
	ssize_t r = opends_sync_read(fh, gpu, n, (off_t)dev_off, 0);
	if (r > 0)
		cuda_buf_to_host(host, gpu, (size_t)r);
	return r;
}

static inline ssize_t
raw_write(opends_handle_t fh, void *gpu, const unsigned char *host, size_t n,
          uint64_t dev_off)
{
	cuda_buf_from_host(gpu, host, n);
	return opends_sync_write(fh, gpu, n, (off_t)dev_off, 0);
}

#endif /* TEST_RAW_BLOCK_H_ */
