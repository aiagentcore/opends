/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_raw_block_aisio - Raw block mode on the aisio backend.
 *
 * Registers the whole-disk block device qublk exports (/dev/ublkbN) and
 * drives P2P reads and writes at namespace byte offsets, with no file
 * system in the path. The oracle is the sync-read pattern file: FIEMAP on
 * the mounted XFS gives the device byte offset of each of its extents, and
 * a raw read at that offset must return the same bytes the kernel wrote
 * there. The write legs (sync and stream-ordered) overwrite one page of that
 * file's own extent through the raw handle, read it back, then restore the
 * pattern, so the file system and the page cache end the test consistent
 * with the disk.
 *
 * Also checks the gate: with OPENDS_AISIO_RAW_BLOCK unset a block device fd
 * is refused at registration, and with it set a block device other than the
 * ublk disk is refused too. The suite runs the test under
 * tasks/steps/with_loop_dev.sh, which names a loop device for that check in
 * OPENDS_TEST_FOREIGN_BLKDEV; without the variable the check is skipped.
 *
 * usage: test_raw_block_aisio <raw_block_dev> <pattern_path_on_mount>
 */
#define _GNU_SOURCE

#include "read_pattern.h"
#include "test_aisio_homi.h"
#include "test_cuda_common.h"
#include "test_fiemap.h"
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
#include <unistd.h>

#define MAX_EXTENTS 64

/* The pattern file's extents, each plain written data and clamped to the
 * file size. */
static int
pattern_extents(int fd, struct test_extent *out, uint32_t *n_out)
{
	bool complete;
	int n = test_fiemap(fd, out, MAX_EXTENTS, &complete);
	if (n < 0)
		return n;
	if (!complete)
		return -E2BIG;
	for (int i = 0; i < n; i++) {
		if (!out[i].plain)
			return -ENOTSUP;
		if (out[i].file_off + out[i].len > FILE_SIZE)
			out[i].len = FILE_SIZE - out[i].file_off;
	}
	*n_out = (uint32_t)n;
	return 0;
}

/* Put the pattern back on the first page of extent e through the raw handle
 * and read it back. Returns 0 when the page holds the pattern again. */
static int
restore_page(opends_handle_t fh, void *gpu, unsigned char *host,
             const struct test_extent *e)
{
	unsigned char *img = malloc(PAGE);
	if (!img)
		return -1;
	for (size_t i = 0; i < PAGE; i++)
		img[i] = pattern_byte((off_t)(e->file_off + i));
	ssize_t w = raw_write(fh, gpu, img, PAGE, e->dev_off);
	ssize_t r = raw_read(fh, gpu, host, PAGE, e->dev_off);
	free(img);
	if (w != PAGE || r != PAGE || check_pattern(host, e->file_off, PAGE)) {
		fprintf(stderr,
		        "restore failed (write %zd, read %zd); "
		        "rerun test_sync_read_prep\n",
		        w, r);
		return -1;
	}
	return 0;
}

int
main(int argc, char **argv)
{
	if (argc != 3) {
		fprintf(stderr, "usage: %s <raw_block_dev> <pattern_path>\n",
		        argv[0]);
		return 1;
	}
	const char *raw_path = argv[1];
	const char *pattern_path = argv[2];
	int failed = 0;

	fprintf(stderr, "raw block mode tests (aisio backend, %s)\n", raw_path);

	/* ---- 1. Gate: raw mode off, block device fd refused. ---- */
	unsetenv(ENV_RAW_BLOCK);
	struct aisio_homi a;
	if (aisio_homi_setup_flags(pattern_path, O_RDONLY | O_DIRECT, &a) < 0)
		return 1;

	int raw_fd = open(raw_path, O_RDWR | O_DIRECT);
	if (raw_fd < 0) {
		fprintf(stderr, "open(%s): %s\n", raw_path, strerror(errno));
		aisio_homi_teardown(&a);
		return 1;
	}
	opends_handle_t raw_fh = NULL;
	opends_error_t err = opends_handle_register(&raw_fh, raw_fd);
	if (err.err != OPENDS_INVALID_VALUE) {
		fprintf(stderr,
		        "gate: block fd not refused with %s unset "
		        "(err %d)\n",
		        ENV_RAW_BLOCK, err.err);
		if (err.err == OPENDS_SUCCESS)
			opends_handle_deregister(raw_fh);
		failed++;
	} else {
		fprintf(stderr, "gate: block fd refused with %s unset ok\n",
		        ENV_RAW_BLOCK);
	}
	aisio_homi_teardown(&a);

	/* ---- 2. Raw mode on: fresh driver, raw handle beside a file. ---- */
	setenv(ENV_RAW_BLOCK, "1", 1);
	if (aisio_homi_setup_flags(pattern_path, O_RDONLY | O_DIRECT, &a) < 0) {
		close(raw_fd);
		return 1;
	}
	raw_fh = NULL;
	err = opends_handle_register(&raw_fh, raw_fd);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "register raw fd: %s\n",
		        opends_op_status_error(err.err));
		aisio_homi_teardown(&a);
		close(raw_fd);
		return 1;
	}

	/* ---- 2b. Raw mode on: a block device other than the ublk disk is
	 * refused, since its offsets are not the namespace's LBAs. ---- */
	const char *foreign = getenv("OPENDS_TEST_FOREIGN_BLKDEV");
	int foreign_fd = foreign ? open(foreign, O_RDWR | O_DIRECT) : -1;
	if (!foreign) {
		fprintf(stderr, "foreign block device: skipped "
		                "(OPENDS_TEST_FOREIGN_BLKDEV unset)\n");
	} else if (foreign_fd < 0) {
		fprintf(stderr, "open(%s): %s\n", foreign, strerror(errno));
		failed++;
	} else {
		opends_handle_t ffh = NULL;
		err = opends_handle_register(&ffh, foreign_fd);
		if (err.err != OPENDS_INVALID_VALUE) {
			fprintf(stderr,
			        "foreign block device %s not refused "
			        "(err %d)\n",
			        foreign, err.err);
			if (err.err == OPENDS_SUCCESS)
				opends_handle_deregister(ffh);
			failed++;
		} else {
			fprintf(stderr, "foreign block device refused ok\n");
		}
		close(foreign_fd);
	}

	uint64_t dev_nbytes = 0;
	if (ioctl(raw_fd, BLKGETSIZE64, &dev_nbytes) < 0) {
		fprintf(stderr, "BLKGETSIZE64: %s\n", strerror(errno));
		failed++;
	}

	struct test_extent ext[MAX_EXTENTS];
	uint32_t n_ext = 0;
	int frc = pattern_extents(a.fd, ext, &n_ext);
	if (frc < 0) {
		fprintf(stderr, "FIEMAP(%s): %s\n", pattern_path,
		        strerror(-frc));
		opends_handle_deregister(raw_fh);
		aisio_homi_teardown(&a);
		close(raw_fd);
		return 1;
	}
	fprintf(stderr,
	        "pattern file has %u extent(s); first at device "
	        "offset %" PRIu64 "\n",
	        n_ext, ext[0].dev_off);

	void *gpu = cuda_alloc_acquire(FILE_SIZE);
	unsigned char *host = malloc(FILE_SIZE);
	if (!gpu || !host) {
		fprintf(stderr, "buffer allocation failed\n");
		return 1;
	}

	/* ---- 3. Raw read of every extent matches the pattern. ---- */
	for (uint32_t i = 0; i < n_ext; i++) {
		ssize_t r =
		        raw_read(raw_fh, gpu, host, ext[i].len, ext[i].dev_off);
		size_t bad = r == (ssize_t)ext[i].len
		                     ? check_pattern(host, ext[i].file_off,
		                                     ext[i].len)
		                     : ext[i].len;
		if (bad) {
			fprintf(stderr,
			        "extent %u: raw read returned %zd, %zu bad "
			        "bytes\n",
			        i, r, bad);
			failed++;
		}
	}
	fprintf(stderr, "raw read of %u extent(s) %s\n", n_ext,
	        failed ? "FAILED" : "ok");

	/* ---- 4. Unaligned raw read (head and tail bounce). The start is
	 * off the LBA but on a 4-byte boundary, as the aisio read rules in
	 * opends.h require of a read that spans whole LBAs. ---- */
	{
		size_t n = PAGE - 8;
		ssize_t r = raw_read(raw_fh, gpu, host, n, ext[0].dev_off + 4);
		size_t bad =
		        r == (ssize_t)n
		                ? check_pattern(host, ext[0].file_off + 4, n)
		                : n;
		if (bad) {
			fprintf(stderr,
			        "unaligned raw read: returned %zd, %zu "
			        "bad bytes\n",
			        r, bad);
			failed++;
		} else {
			fprintf(stderr, "unaligned raw read ok\n");
		}
	}

	/* ---- 5. Read into a cudaMalloc buffer (opends_buf_register). ---- */
	{
		void *reg = cuda_register_acquire(PAGE);
		if (!reg) {
			failed++;
		} else {
			ssize_t r = raw_read(raw_fh, reg, host, PAGE,
			                     ext[0].dev_off);
			size_t bad =
			        r == PAGE ? check_pattern(host, ext[0].file_off,
			                                  PAGE)
			                  : PAGE;
			if (bad) {
				fprintf(stderr,
				        "register-mode raw read: "
				        "returned %zd, %zu bad bytes\n",
				        r, bad);
				failed++;
			} else {
				fprintf(stderr, "register-mode raw read ok\n");
			}
			cuda_register_release(reg);
		}
	}

	/* ---- 6. Aligned raw write, read back, restore. ---- */
	{
		unsigned char *inv = malloc(PAGE);
		for (size_t i = 0; i < PAGE; i++)
			inv[i] = (unsigned char)~pattern_byte(
			        (off_t)(ext[0].file_off + i));

		ssize_t w = raw_write(raw_fh, gpu, inv, PAGE, ext[0].dev_off);
		if (w != PAGE) {
			fprintf(stderr, "raw write returned %zd\n", w);
			failed++;
		} else {
			ssize_t r = raw_read(raw_fh, gpu, host, PAGE,
			                     ext[0].dev_off);
			if (r != PAGE || memcmp(host, inv, PAGE) != 0) {
				fprintf(stderr,
				        "raw write read-back mismatch "
				        "(read %zd)\n",
				        r);
				failed++;
			} else {
				fprintf(stderr, "raw write + read-back ok\n");
			}
		}

		/* Restore, whatever happened above, so the pattern file is
		 * intact for the rest of the suite. */
		if (restore_page(raw_fh, gpu, host, &ext[0]) < 0)
			failed++;
		else
			fprintf(stderr, "pattern restored ok\n");
		free(inv);
	}

	/* ---- 6b. Stream-ordered raw I/O on the first extent: a stream read
	 * of the whole extent, then a stream write of one page followed on the
	 * same stream by a read of it, waited for once. ---- */
	{
		CUstream stream;
		if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) !=
		            CUDA_SUCCESS ||
		    opends_stream_register(stream, 0).err != OPENDS_SUCCESS) {
			fprintf(stderr, "stream setup failed\n");
			failed++;
			goto stream_done;
		}

		ssize_t r = raw_stream_read(raw_fh, stream, gpu, host,
		                            ext[0].len, ext[0].dev_off);
		size_t bad = r == (ssize_t)ext[0].len
		                     ? check_pattern(host, ext[0].file_off,
		                                     ext[0].len)
		                     : ext[0].len;
		if (bad) {
			fprintf(stderr,
			        "stream raw read: returned %zd, %zu bad "
			        "bytes\n",
			        r, bad);
			failed++;
		} else {
			fprintf(stderr, "stream raw read ok\n");
		}

		unsigned char *inv = malloc(PAGE);
		for (size_t i = 0; i < PAGE; i++)
			inv[i] = (unsigned char)~pattern_byte(
			        (off_t)(ext[0].file_off + i));
		cuda_buf_from_host(gpu, inv, PAGE);
		cuda_buf_zero((uint8_t *)gpu + PAGE, PAGE);
		size_t wsz = PAGE, rsz = PAGE;
		off_t wfo = (off_t)ext[0].dev_off, rfo = wfo;
		off_t wbo = 0, rbo = PAGE;
		ssize_t wb = -1, rb = -1;
		opends_error_t werr = opends_stream_write(
		        raw_fh, gpu, &wsz, &wfo, &wbo, &wb, stream);
		opends_error_t rerr = opends_stream_read(
		        raw_fh, gpu, &rsz, &rfo, &rbo, &rb, stream);
		int waited = raw_stream_wait(stream, 10.0, "stream write+read");
		cuda_buf_to_host(host, (uint8_t *)gpu + PAGE, PAGE);
		if (werr.err != OPENDS_SUCCESS || rerr.err != OPENDS_SUCCESS ||
		    waited < 0 || wb != PAGE || rb != PAGE ||
		    memcmp(host, inv, PAGE) != 0) {
			fprintf(stderr,
			        "stream raw write then read: submit %d/%d, "
			        "wrote %zd, read %zd\n",
			        werr.err, rerr.err, wb, rb);
			failed++;
		} else {
			fprintf(stderr,
			        "stream raw write + in-order read ok\n");
		}
		free(inv);

		if (restore_page(raw_fh, gpu, host, &ext[0]) < 0)
			failed++;
		opends_stream_deregister(stream);
		cuStreamDestroy(stream);
	}
stream_done:

	/* ---- 7. Unaligned raw writes are refused, nothing written. ---- */
	{
		/* Start off an LBA boundary. */
		cuda_buf_zero(gpu, PAGE);
		ssize_t w = opends_sync_write(raw_fh, gpu, PAGE,
		                              (off_t)(ext[0].dev_off + 1), 0);
		ssize_t r = raw_read(raw_fh, gpu, host, PAGE, ext[0].dev_off);
		if (w != -(ssize_t)OPENDS_INVALID_VALUE || r != PAGE ||
		    check_pattern(host, ext[0].file_off, PAGE)) {
			fprintf(stderr,
			        "unaligned-start raw write: returned "
			        "%zd (want %d), pattern intact=%s\n",
			        w, -OPENDS_INVALID_VALUE,
			        r == PAGE ? "yes" : "no");
			failed++;
			if (restore_page(raw_fh, gpu, host, &ext[0]) < 0)
				failed++;
		} else {
			fprintf(stderr,
			        "unaligned-start raw write refused ok\n");
		}

		/* Start on an LBA boundary, end off one. The whole op must be
		 * refused before anything reaches the device: the first page
		 * must still hold the pattern afterwards. */
		size_t n = PAGE + 1;
		if (ext[0].len >= n) {
			cuda_buf_zero(gpu, n);
			w = opends_sync_write(raw_fh, gpu, n,
			                      (off_t)ext[0].dev_off, 0);
			r = raw_read(raw_fh, gpu, host, PAGE, ext[0].dev_off);
			size_t bad =
			        r == PAGE ? check_pattern(host, ext[0].file_off,
			                                  PAGE)
			                  : PAGE;
			if (w != -(ssize_t)OPENDS_INVALID_VALUE || bad) {
				fprintf(stderr,
				        "unaligned-end raw write: returned %zd "
				        "(want %d), %zu byte(s) of the first "
				        "page changed\n",
				        w, -OPENDS_INVALID_VALUE, bad);
				failed++;
				if (bad && restore_page(raw_fh, gpu, host,
				                        &ext[0]) < 0)
					failed++;
			} else {
				fprintf(stderr,
				        "unaligned-end raw write refused ok\n");
			}
		}
	}

	/* ---- 8. Reads at and past the end of the device. ---- */
	if (dev_nbytes >= 2 * PAGE) {
		ssize_t r = raw_read(raw_fh, gpu, host, 2 * PAGE,
		                     dev_nbytes - PAGE);
		ssize_t z = opends_sync_read(raw_fh, gpu, PAGE,
		                             (off_t)dev_nbytes, 0);
		if (r != PAGE || z != 0) {
			fprintf(stderr,
			        "end of device: read %zd (want %d), "
			        "past end %zd (want 0)\n",
			        r, PAGE, z);
			failed++;
		} else {
			fprintf(stderr, "end-of-device reads ok\n");
		}
	}

	cuda_alloc_release(gpu);
	free(host);
	opends_handle_deregister(raw_fh);
	aisio_homi_teardown(&a);
	close(raw_fd);

	if (failed)
		fprintf(stderr, "%d test(s) failed\n", failed);
	else
		fprintf(stderr, "all ok\n");
	return failed ? 1 : 0;
}
