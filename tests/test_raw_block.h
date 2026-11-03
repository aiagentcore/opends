/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_raw_block.h - Helpers shared by the raw block mode tests: a raw
 * read or write through a registered block device handle (sync or
 * stream-ordered), staged through a GPU buffer, and a pattern check.
 */
#ifndef TEST_RAW_BLOCK_H_
#define TEST_RAW_BLOCK_H_

#include "opends.h"
#include "read_pattern.h"
#include "test_cuda_common.h"

#include <cuda.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/types.h>
#include <time.h>

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

/* Wait for a stream to drain, bounded by a wall-clock timeout so a stuck
 * backend fails the test instead of hanging the runner. Returns 0 when the
 * stream is idle, -1 after a message on timeout or driver error. */
static inline int
raw_stream_wait(CUstream stream, double timeout_s, const char *label)
{
	struct timespec t0, now;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	for (;;) {
		CUresult cr = cuStreamQuery(stream);
		if (cr == CUDA_SUCCESS)
			return 0;
		if (cr != CUDA_ERROR_NOT_READY) {
			fprintf(stderr, "  %s: cuStreamQuery: %d\n", label,
			        (int)cr);
			return -1;
		}
		clock_gettime(CLOCK_MONOTONIC, &now);
		double elapsed = (now.tv_sec - t0.tv_sec) +
		                 (now.tv_nsec - t0.tv_nsec) * 1e-9;
		if (elapsed >= timeout_s) {
			fprintf(stderr, "  %s: timed out after %.1fs\n", label,
			        elapsed);
			return -1;
		}
		struct timespec sl = {0, 1000000};
		nanosleep(&sl, NULL);
	}
}

/* Stream read of n bytes at device offset dev_off into gpu, waited for on
 * the stream, then copied to host. Returns the backend's result (bytes or
 * -error), or -OPENDS_INTERNAL_ERROR when the submit or the wait failed. */
static inline ssize_t
raw_stream_read(opends_handle_t fh, CUstream stream, void *gpu,
                unsigned char *host, size_t n, uint64_t dev_off)
{
	size_t size = n;
	off_t foff = (off_t)dev_off;
	off_t boff = 0;
	ssize_t bytes = -1;
	cuda_buf_zero(gpu, n);
	opends_error_t err = opends_stream_read(fh, gpu, &size, &foff, &boff,
	                                        &bytes, stream);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  stream_read submit: %s\n",
		        opends_op_status_error(err.err));
		return -(ssize_t)OPENDS_INTERNAL_ERROR;
	}
	if (raw_stream_wait(stream, 10.0, "stream_read") < 0)
		return -(ssize_t)OPENDS_INTERNAL_ERROR;
	if (bytes > 0)
		cuda_buf_to_host(host, gpu, (size_t)bytes);
	return bytes;
}

static inline ssize_t
raw_stream_write(opends_handle_t fh, CUstream stream, void *gpu,
                 const unsigned char *host, size_t n, uint64_t dev_off)
{
	size_t size = n;
	off_t foff = (off_t)dev_off;
	off_t boff = 0;
	ssize_t bytes = -1;
	cuda_buf_from_host(gpu, host, n);
	opends_error_t err = opends_stream_write(fh, gpu, &size, &foff, &boff,
	                                         &bytes, stream);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "  stream_write submit: %s\n",
		        opends_op_status_error(err.err));
		return -(ssize_t)OPENDS_INTERNAL_ERROR;
	}
	if (raw_stream_wait(stream, 10.0, "stream_write") < 0)
		return -(ssize_t)OPENDS_INTERNAL_ERROR;
	return bytes;
}

#endif /* TEST_RAW_BLOCK_H_ */
