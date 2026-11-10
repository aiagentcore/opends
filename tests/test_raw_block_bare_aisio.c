/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_raw_block_bare_aisio - Raw block mode with no file system in the
 * stack: only homi and qublk are up, nothing is mounted over the ublk device
 * and no xal-server publishes an index. This is the uGDS-style use of the
 * backend, a process that owns the block device and lays it out itself.
 *
 * The harness (tasks/test.yaml) writes a scratch file, records the byte range
 * of its first extent on the device, unmounts the file system and passes the
 * range here. The file is removed only after the test, so the file system
 * still owns those blocks and nothing else is in them. The test writes the
 * sync-read pattern into the range through the raw handle and reads it back
 * with the sync, async, batch and stream families. The gate, the oracle against
 * kernel-written data and the alignment rules are in test_raw_block_aisio.
 *
 * It also checks that the driver opens without an index, and that a regular
 * file (this binary, on an unindexed file system) is refused at registration
 * with OPENDS_FS_SETUP_ERROR, the second time without trying the index again.
 * The test names an index of its own (OPENDS_XAL_SHM), whatever the caller
 * set.
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
#include <linux/fs.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define REGION_PAGES 8
#define REGION_BYTES ((size_t)PAGE * REGION_PAGES)

/* An index name with a malformed state region. Attaching to it fails at once;
 * a missing index would be waited for, for a minute. */
#define BAD_INDEX "/opends_raw_bare_bad_index"

/* One batch of two one-page ops on consecutive pages at dev_off, from or into
 * the first two pages of gpu. Returns 1 when both completed in full. */
static int
batch_two_pages(opends_handle_t fh, void *gpu, uint64_t dev_off,
                opends_opcode_t opcode)
{
	opends_batch_handle_t batch = NULL;
	opends_io_params_t io[2];
	opends_io_events_t ev[2];
	unsigned nr = 2;

	if (opends_batch_setup(&batch, 2).err != OPENDS_SUCCESS)
		return 0;
	memset(io, 0, sizeof(io));
	for (int i = 0; i < 2; i++) {
		io[i].mode = OPENDS_BATCH;
		io[i].fh = fh;
		io[i].opcode = opcode;
		io[i].cookie = &io[i];
		io[i].u.batch.dev_ptr_base = gpu;
		io[i].u.batch.dev_ptr_offset = (off_t)i * PAGE;
		io[i].u.batch.file_offset = (off_t)(dev_off + i * PAGE);
		io[i].u.batch.size = PAGE;
	}
	int ok = opends_batch_submit(batch, 2, io, 0).err == OPENDS_SUCCESS &&
	         opends_batch_get_status(batch, 2, &nr, ev, NULL).err ==
	                 OPENDS_SUCCESS &&
	         nr == 2 && ev[0].status == OPENDS_COMPLETE &&
	         ev[1].status == OPENDS_COMPLETE && ev[0].ret == PAGE &&
	         ev[1].ret == PAGE;
	opends_batch_destroy(batch);
	return ok;
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
	char *off_end, *len_end;
	uint64_t dev_off = strtoull(argv[2], &off_end, 0);
	uint64_t nbytes = strtoull(argv[3], &len_end, 0);
	int failed = 0;

	fprintf(stderr, "raw block mode, bare device (aisio backend, %s)\n",
	        raw_path);
	/* The test writes to this range. Offset 0 is no file's extent (the
	 * file system's superblock is there), and it is what a bad argument
	 * parses to. */
	if (!*argv[2] || *off_end || !*argv[3] || *len_end || !dev_off ||
	    nbytes < REGION_BYTES || (dev_off & (PAGE - 1))) {
		fprintf(stderr,
		        "need a page-aligned range of at least %zu bytes past "
		        "offset 0, got \"%s\" at \"%s\"\n",
		        REGION_BYTES, argv[3], argv[2]);
		return 1;
	}

	CUcontext cuctx;
	if (cuda_ctx_create(&cuctx) < 0) {
		fprintf(stderr, "cuCtxCreate failed\n");
		return 1;
	}

	/* ---- 1. The driver opens with no index published. ---- */
	int sfd = shm_open(BAD_INDEX "_state", O_CREAT | O_RDWR, 0600);
	if (sfd < 0 || write(sfd, "x", 1) != 1) {
		fprintf(stderr, "shm_open(%s): %s\n", BAD_INDEX "_state",
		        strerror(errno));
		cuCtxDestroy(cuctx);
		return 1;
	}
	close(sfd);
	setenv("OPENDS_XAL_SHM", BAD_INDEX, 1);
	setenv(ENV_RAW_BLOCK, "1", 1);
	opends_error_t err = opends_driver_open();
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "driver_open without xal-server: %s\n",
		        opends_op_status_error(err.err));
		shm_unlink(BAD_INDEX "_state");
		cuCtxDestroy(cuctx);
		return 1;
	}
	fprintf(stderr, "driver open without an index ok\n");

	/* ---- 2. A regular file is refused: no index to resolve it in. The
	 * failed attach is final: with the malformed index gone, a second
	 * registration that tried again would wait a minute for an index. It
	 * must be refused at once instead. ---- */
	{
		int ffd = open(argv[0], O_RDONLY | O_DIRECT);
		opends_handle_t ffh = NULL;
		if (ffd < 0) {
			fprintf(stderr, "open(%s, O_DIRECT): %s\n", argv[0],
			        strerror(errno));
			failed++;
		} else {
			err = opends_handle_register(&ffh, ffd);
			shm_unlink(BAD_INDEX "_state");
			time_t t0 = time(NULL);
			opends_handle_t ffh2 = NULL;
			opends_error_t err2 =
			        opends_handle_register(&ffh2, ffd);
			if (err.err != OPENDS_FS_SETUP_ERROR ||
			    err2.err != OPENDS_FS_SETUP_ERROR ||
			    time(NULL) - t0 > 5) {
				fprintf(stderr,
				        "regular file without an index: err "
				        "%d, then err %d after %lds (want %d "
				        "twice, the second at once)\n",
				        err.err, err2.err,
				        (long)(time(NULL) - t0),
				        OPENDS_FS_SETUP_ERROR);
				if (err.err == OPENDS_SUCCESS)
					opends_handle_deregister(ffh);
				if (err2.err == OPENDS_SUCCESS)
					opends_handle_deregister(ffh2);
				failed++;
			} else {
				fprintf(stderr, "regular file refused without "
				                "an index ok, twice\n");
			}
			close(ffd);
		}
	}
	shm_unlink(BAD_INDEX "_state");

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

	/* ---- 6. Batch write of two pages, batch read of them. ---- */
	{
		for (size_t i = 0; i < 2 * PAGE; i++)
			host[i] = (unsigned char)~img[3 * PAGE + i];
		cuda_buf_from_host(gpu, host, 2 * PAGE);
		int wok = batch_two_pages(fh, gpu, dev_off + 3 * PAGE,
		                          OPENDS_WRITE);
		cuda_buf_zero(gpu, 2 * PAGE);
		int rok = batch_two_pages(fh, gpu, dev_off + 3 * PAGE,
		                          OPENDS_READ);
		cuda_buf_to_host(host, gpu, 2 * PAGE);
		int mismatch = 0;
		for (size_t i = 0; i < 2 * PAGE; i++)
			if (host[i] != (unsigned char)~img[3 * PAGE + i])
				mismatch++;
		if (!wok || !rok || mismatch) {
			fprintf(stderr,
			        "batch write %s, batch read %s, %d "
			        "mismatches\n",
			        wok ? "ok" : "failed", rok ? "ok" : "failed",
			        mismatch);
			failed++;
		} else {
			fprintf(stderr, "batch write + batch read ok\n");
		}
		/* put the pattern back on those pages */
		raw_write(fh, gpu, img + 3 * PAGE, 2 * PAGE,
		          dev_off + 3 * PAGE);
	}

	/* ---- 7. Stream-ordered I/O: a stream read, a partial-tail stream
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

		/* Ends off an LBA: refused, and page 6 keeps the inverse
		 * instead of the pattern this write carries. */
		ssize_t w = raw_stream_write(fh, stream, gpu, img + 6 * PAGE,
		                             PAGE + 1, dev_off + 6 * PAGE);
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

	/* ---- 8. A write across the end of the device is short. It writes
	 * the last page back as it is, so the device does not change. ---- */
	{
		uint64_t dev_nbytes = 0;
		ssize_t r = -1, w = -1;
		if (ioctl(raw_fd, BLKGETSIZE64, &dev_nbytes) == 0 &&
		    dev_nbytes >= PAGE)
			r = raw_read(fh, gpu, host, PAGE, dev_nbytes - PAGE);
		if (r == PAGE)
			w = opends_sync_write(fh, gpu, 2 * PAGE,
			                      (off_t)(dev_nbytes - PAGE), 0);
		if (w != PAGE) {
			fprintf(stderr,
			        "write across the end: read %zd, wrote %zd "
			        "(want %d)\n",
			        r, w, PAGE);
			failed++;
		} else {
			fprintf(stderr, "write across the end short ok\n");
		}
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
