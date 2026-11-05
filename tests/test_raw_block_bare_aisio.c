/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_raw_block_bare_aisio - Raw block mode with no file system in the
 * stack: only homi and qublk are up, nothing is mounted over the ublk device
 * and no xal-server publishes an index. This is the uGDS-style use of the
 * backend, a process that owns the block device and lays it out itself.
 *
 * The harness (tasks/test.yaml) records a free byte range of the device
 * before it unmounts the file system (the blocks the suite's pattern file
 * used, freed when clean_test_dir deleted it) and passes it here. The test
 * writes the sync-read pattern into that range through the raw handle and
 * reads it back with the sync, async, batch and stream families, checks the
 * alignment rules, and zeroes the range when done.
 *
 * It also checks that the driver opens without an index, and that a regular
 * file (this binary, on an unindexed file system) is refused at registration
 * with OPENDS_FS_SETUP_ERROR. That refusal waits a minute for an index to
 * appear before giving up.
 *
 * usage: test_raw_block_bare_aisio <raw_block_dev> <dev_off> <nbytes>
 */
#define _GNU_SOURCE

#include "read_pattern.h"
#include "test_cuda_common.h"
#include "test_raw_block.h"

#include <cuda.h>

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define REGION_PAGES 8
#define REGION_BYTES ((size_t)PAGE * REGION_PAGES)

static int
cuda_ctx_create(CUcontext *ctx)
{
	CUdevice dev;
	cuInit(0);
	cuDeviceGet(&dev, 0);
#if CUDA_VERSION >= 13000
	return cuCtxCreate(ctx, NULL, 0, dev) == CUDA_SUCCESS ? 0 : -1;
#else
	return cuCtxCreate(ctx, 0, dev) == CUDA_SUCCESS ? 0 : -1;
#endif
}

int
main(int argc, char **argv)
{
	if (argc != 4) {
		fprintf(stderr,
		        "usage: %s <raw_block_dev> <dev_off> <nbytes>\n",
		        argv[0]);
		return 1;
	}
	const char *raw_path = argv[1];
	uint64_t dev_off = strtoull(argv[2], NULL, 0);
	uint64_t nbytes = strtoull(argv[3], NULL, 0);
	int failed = 0;

	fprintf(stderr, "raw block mode, bare device (aisio backend, %s)\n",
	        raw_path);
	if (nbytes < REGION_BYTES || (dev_off & (PAGE - 1))) {
		fprintf(stderr,
		        "need a page-aligned range of at least %zu bytes, got "
		        "%" PRIu64 " at %" PRIu64 "\n",
		        REGION_BYTES, nbytes, dev_off);
		return 1;
	}

	CUcontext cuctx;
	if (cuda_ctx_create(&cuctx) < 0) {
		fprintf(stderr, "cuCtxCreate failed\n");
		return 1;
	}

	/* ---- 1. The driver opens with no index published. ---- */
	setenv(ENV_RAW_BLOCK, "1", 1);
	opends_error_t err = opends_driver_open();
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "driver_open without xal-server: %s\n",
		        opends_op_status_error(err.err));
		cuCtxDestroy(cuctx);
		return 1;
	}
	fprintf(stderr, "driver open without an index ok\n");

	/* ---- 2. A regular file is refused: no index to resolve it in. ---- */
	{
		int ffd = open(argv[0], O_RDONLY | O_DIRECT);
		opends_handle_t ffh = NULL;
		if (ffd < 0) {
			fprintf(stderr, "open(%s, O_DIRECT): %s\n", argv[0],
			        strerror(errno));
			failed++;
		} else {
			fprintf(stderr, "registering a regular file (waits for "
			                "an index for a minute)...\n");
			err = opends_handle_register(&ffh, ffd);
			if (err.err != OPENDS_FS_SETUP_ERROR) {
				fprintf(stderr,
				        "regular file not refused without an "
				        "index (err %d)\n",
				        err.err);
				if (err.err == OPENDS_SUCCESS)
					opends_handle_deregister(ffh);
				failed++;
			} else {
				fprintf(stderr, "regular file refused without "
				                "an index ok\n");
			}
			close(ffd);
		}
	}

	/* ---- 3. The raw handle registers. ---- */
	int raw_fd = open(raw_path, O_RDWR | O_DIRECT);
	if (raw_fd < 0) {
		fprintf(stderr, "open(%s): %s\n", raw_path, strerror(errno));
		opends_driver_close();
		cuCtxDestroy(cuctx);
		return 1;
	}
	opends_handle_t fh = NULL;
	err = opends_handle_register(&fh, raw_fd);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "register raw fd: %s\n",
		        opends_op_status_error(err.err));
		close(raw_fd);
		opends_driver_close();
		cuCtxDestroy(cuctx);
		return 1;
	}

	void *gpu = cuda_alloc_acquire(REGION_BYTES);
	unsigned char *host = malloc(REGION_BYTES);
	unsigned char *img = malloc(REGION_BYTES);
	if (!gpu || !host || !img) {
		fprintf(stderr, "buffer allocation failed\n");
		return 1;
	}
	for (size_t i = 0; i < REGION_BYTES; i++)
		img[i] = pattern_byte((off_t)i);

	/* ---- 4. Sync write of the pattern, sync read back. ---- */
	{
		ssize_t w = raw_write(fh, gpu, img, REGION_BYTES, dev_off);
		ssize_t r = raw_read(fh, gpu, host, REGION_BYTES, dev_off);
		size_t bad = r == (ssize_t)REGION_BYTES
		                     ? check_pattern(host, 0, REGION_BYTES)
		                     : REGION_BYTES;
		if (w != (ssize_t)REGION_BYTES || bad) {
			fprintf(stderr,
			        "sync write %zd, read %zd, %zu bad bytes\n", w,
			        r, bad);
			failed++;
		} else {
			fprintf(stderr, "sync write + read-back ok\n");
		}
	}

	/* ---- 5. Async read of one page, async rewrite of another. ---- */
	{
		opends_async_future_t fut;
		cuda_buf_zero(gpu, PAGE);
		err = opends_async_read(fh, gpu, PAGE, (off_t)(dev_off + PAGE),
		                        0, &fut);
		ssize_t r = err.err == OPENDS_SUCCESS ? opends_async_await(&fut)
		                                      : -(ssize_t)err.err;
		if (r == PAGE)
			cuda_buf_to_host(host, gpu, PAGE);
		size_t bad = r == PAGE ? check_pattern(host, PAGE, PAGE) : PAGE;
		if (bad) {
			fprintf(stderr, "async read: %zd, %zu bad bytes\n", r,
			        bad);
			failed++;
		}

		for (size_t i = 0; i < PAGE; i++)
			host[i] = (unsigned char)~img[2 * PAGE + i];
		cuda_buf_from_host(gpu, host, PAGE);
		err = opends_async_write(fh, gpu, PAGE,
		                         (off_t)(dev_off + 2 * PAGE), 0, &fut);
		ssize_t w = err.err == OPENDS_SUCCESS ? opends_async_await(&fut)
		                                      : -(ssize_t)err.err;
		r = raw_read(fh, gpu, host, PAGE, dev_off + 2 * PAGE);
		int mismatch = 0;
		for (size_t i = 0; i < PAGE; i++)
			if (host[i] != (unsigned char)~img[2 * PAGE + i])
				mismatch++;
		if (w != PAGE || r != PAGE || mismatch) {
			fprintf(stderr,
			        "async write %zd, read %zd, %d mismatches\n", w,
			        r, mismatch);
			failed++;
		} else {
			fprintf(stderr, "async read + async write ok\n");
		}
		/* put the pattern back on that page */
		raw_write(fh, gpu, img + 2 * PAGE, PAGE, dev_off + 2 * PAGE);
	}

	/* ---- 6. Batch read of two pages. ---- */
	{
		opends_batch_handle_t batch = NULL;
		err = opends_batch_setup(&batch, 2);
		if (err.err != OPENDS_SUCCESS) {
			fprintf(stderr, "batch_setup: %s\n",
			        opends_op_status_error(err.err));
			failed++;
		} else {
			cuda_buf_zero(gpu, 2 * PAGE);
			opends_io_params_t io[2];
			memset(io, 0, sizeof(io));
			for (int i = 0; i < 2; i++) {
				io[i].mode = OPENDS_BATCH;
				io[i].fh = fh;
				io[i].opcode = OPENDS_READ;
				io[i].cookie = &io[i];
				io[i].u.batch.dev_ptr_base = gpu;
				io[i].u.batch.dev_ptr_offset = (off_t)i * PAGE;
				io[i].u.batch.file_offset =
				        (off_t)(dev_off + (3 + i) * PAGE);
				io[i].u.batch.size = PAGE;
			}
			err = opends_batch_submit(batch, 2, io, 0);
			unsigned nr = 2;
			opends_io_events_t ev[2];
			opends_error_t gerr = opends_batch_get_status(
			        batch, 2, &nr, ev, NULL);
			int ok = err.err == OPENDS_SUCCESS &&
			         gerr.err == OPENDS_SUCCESS && nr == 2 &&
			         ev[0].status == OPENDS_COMPLETE &&
			         ev[1].status == OPENDS_COMPLETE &&
			         ev[0].ret == PAGE && ev[1].ret == PAGE;
			cuda_buf_to_host(host, gpu, 2 * PAGE);
			size_t bad = check_pattern(host, 3 * PAGE, 2 * PAGE);
			if (!ok || bad) {
				fprintf(stderr,
				        "batch read: submit %d, status %d, nr "
				        "%u, "
				        "%zu bad bytes\n",
				        err.err, gerr.err, nr, bad);
				failed++;
			} else {
				fprintf(stderr, "batch read ok\n");
			}
			opends_batch_destroy(batch);
		}
	}

	/* ---- 7. Alignment rules hold on a bare device too. ---- */
	{
		size_t n = PAGE - 8;
		ssize_t r = raw_read(fh, gpu, host, n, dev_off + 4);
		size_t bad = r == (ssize_t)n ? check_pattern(host, 4, n) : n;
		if (bad) {
			fprintf(stderr, "unaligned raw read: %zd, %zu bad\n", r,
			        bad);
			failed++;
		} else {
			fprintf(stderr, "unaligned raw read ok\n");
		}

		cuda_buf_zero(gpu, PAGE + 1);
		ssize_t w =
		        opends_sync_write(fh, gpu, PAGE + 1, (off_t)dev_off, 0);
		r = raw_read(fh, gpu, host, PAGE, dev_off);
		bad = r == PAGE ? check_pattern(host, 0, PAGE) : PAGE;
		if (w != -(ssize_t)OPENDS_INVALID_VALUE || bad) {
			fprintf(stderr,
			        "unaligned-end raw write: %zd (want %d), %zu "
			        "byte(s) changed\n",
			        w, -OPENDS_INVALID_VALUE, bad);
			failed++;
		} else {
			fprintf(stderr, "unaligned-end raw write refused ok\n");
		}
	}

	/* ---- 8. Stream-ordered I/O: a stream read, a partial-tail stream
	 * read (the stream path has its own bounce kernel), then a stream
	 * write followed on the same stream by a read of the page it wrote,
	 * waited for once: the read must see the write. An unaligned-end
	 * stream write is refused through the completion. ---- */
	{
		CUstream stream;
		if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) !=
		            CUDA_SUCCESS ||
		    opends_stream_register(stream, 0).err != OPENDS_SUCCESS) {
			fprintf(stderr, "stream setup failed\n");
			failed++;
			goto stream_done;
		}

		ssize_t r = raw_stream_read(fh, stream, gpu, host, 2 * PAGE,
		                            dev_off + 4 * PAGE);
		size_t bad = r == 2 * PAGE
		                     ? check_pattern(host, 4 * PAGE, 2 * PAGE)
		                     : 2 * PAGE;
		if (bad) {
			fprintf(stderr, "stream read: %zd, %zu bad bytes\n", r,
			        bad);
			failed++;
		} else {
			fprintf(stderr, "stream read ok\n");
		}

		/* A stream read may only be partial at its tail (opends.h): a
		 * tail-bounced read works, a start off the LBA is refused. */
		size_t n = PAGE - 8;
		r = raw_stream_read(fh, stream, gpu, host, n,
		                    dev_off + 4 * PAGE);
		bad = r == (ssize_t)n ? check_pattern(host, 4 * PAGE, n) : n;
		ssize_t u = raw_stream_read(fh, stream, gpu, host, n,
		                            dev_off + 4 * PAGE + 4);
		if (bad || u != -(ssize_t)OPENDS_INVALID_VALUE) {
			fprintf(stderr,
			        "partial-tail stream read: %zd, %zu bad bytes; "
			        "off-LBA start: %zd (want %d)\n",
			        r, bad, u, -OPENDS_INVALID_VALUE);
			failed++;
		} else {
			fprintf(stderr, "partial-tail stream read ok, off-LBA "
			                "start refused ok\n");
		}

		/* Write the inverse of page 6 from slot 0, read page 6 back
		 * into slot 1, both queued before either runs. */
		for (size_t i = 0; i < PAGE; i++)
			host[i] = (unsigned char)~img[6 * PAGE + i];
		cuda_buf_from_host(gpu, host, PAGE);
		cuda_buf_zero((uint8_t *)gpu + PAGE, PAGE);
		size_t wsz = PAGE, rsz = PAGE;
		off_t wfo = (off_t)(dev_off + 6 * PAGE), rfo = wfo;
		off_t wbo = 0, rbo = PAGE;
		ssize_t wb = -1, rb = -1;
		opends_error_t werr = opends_stream_write(fh, gpu, &wsz, &wfo,
		                                          &wbo, &wb, stream);
		opends_error_t rerr = opends_stream_read(fh, gpu, &rsz, &rfo,
		                                         &rbo, &rb, stream);
		int waited = raw_stream_wait(stream, 10.0, "stream write+read");
		cuda_buf_to_host(host, (uint8_t *)gpu + PAGE, PAGE);
		int mismatch = 0;
		for (size_t i = 0; i < PAGE; i++)
			if (host[i] != (unsigned char)~img[6 * PAGE + i])
				mismatch++;
		if (werr.err != OPENDS_SUCCESS || rerr.err != OPENDS_SUCCESS ||
		    waited < 0 || wb != PAGE || rb != PAGE || mismatch) {
			fprintf(stderr,
			        "stream write then read: submit %d/%d, wrote "
			        "%zd, read %zd, %d mismatches\n",
			        werr.err, rerr.err, wb, rb, mismatch);
			failed++;
		} else {
			fprintf(stderr, "stream write + in-order read ok\n");
		}

		/* Ends off an LBA: refused, and page 6 keeps the inverse. */
		cuda_buf_zero(gpu, PAGE + 1);
		ssize_t w = raw_stream_write(fh, stream, gpu, host, PAGE + 1,
		                             dev_off + 6 * PAGE);
		r = raw_read(fh, gpu, host, PAGE, dev_off + 6 * PAGE);
		mismatch = 0;
		for (size_t i = 0; i < PAGE; i++)
			if (host[i] != (unsigned char)~img[6 * PAGE + i])
				mismatch++;
		if (w != -(ssize_t)OPENDS_INVALID_VALUE || r != PAGE ||
		    mismatch) {
			fprintf(stderr,
			        "unaligned-end stream write: %zd (want %d), "
			        "read %zd, %d byte(s) changed\n",
			        w, -OPENDS_INVALID_VALUE, r, mismatch);
			failed++;
		} else {
			fprintf(stderr,
			        "unaligned-end stream write refused ok\n");
		}

		/* put the pattern back on page 6 */
		raw_write(fh, gpu, img + 6 * PAGE, PAGE, dev_off + 6 * PAGE);
		opends_stream_deregister(stream);
		cuStreamDestroy(stream);
	}
stream_done:

	/* ---- 9. Leave the range zeroed. ---- */
	memset(img, 0, REGION_BYTES);
	if (raw_write(fh, gpu, img, REGION_BYTES, dev_off) !=
	    (ssize_t)REGION_BYTES) {
		fprintf(stderr, "zeroing the range failed\n");
		failed++;
	}

	cuda_alloc_release(gpu);
	free(host);
	free(img);
	opends_handle_deregister(fh);
	close(raw_fd);
	opends_driver_close();
	cuCtxDestroy(cuctx);

	if (failed)
		fprintf(stderr, "%d test(s) failed\n", failed);
	else
		fprintf(stderr, "all ok\n");
	return failed ? 1 : 0;
}
