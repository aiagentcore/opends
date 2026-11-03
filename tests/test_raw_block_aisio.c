/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * test_raw_block_aisio - Raw block mode on the aisio backend.
 *
 * Registers the whole-disk block device qublk exports (/dev/ublkbN) and
 * drives P2P reads and writes at namespace byte offsets, with no file
 * system in the path. The oracle is the sync-read pattern file: FIEMAP on
 * the mounted XFS gives the device byte offset of each of its extents, and
 * a raw read at that offset must return the same bytes the kernel wrote
 * there. The write leg overwrites one page of that file's own extent
 * through the raw handle, reads it back, then restores the pattern, so the
 * file system and the page cache end the test consistent with the disk.
 *
 * Also checks the gate: with OPENDS_AISIO_RAW_BLOCK unset a block device fd
 * is refused at registration.
 *
 * usage: test_raw_block_aisio <raw_block_dev> <pattern_path_on_mount>
 */
#define _GNU_SOURCE

#include "read_pattern.h"
#include "test_aisio_homi.h"
#include "test_cuda_common.h"

#include <cuda.h>

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define ENV_RAW_BLOCK "OPENDS_AISIO_RAW_BLOCK"
#define MAX_EXTENTS 64

/* One extent of the pattern file: where it sits in the file and on disk. */
struct ext {
	uint64_t file_off;
	uint64_t dev_off;
	uint64_t len;
};

static int
fiemap_pattern(int fd, struct ext *out, uint32_t *n_out)
{
	size_t sz = sizeof(struct fiemap) +
	            MAX_EXTENTS * sizeof(struct fiemap_extent);
	struct fiemap *fm = calloc(1, sz);
	if (!fm)
		return -ENOMEM;
	fm->fm_start = 0;
	fm->fm_length = FIEMAP_MAX_OFFSET;
	fm->fm_flags = FIEMAP_FLAG_SYNC;
	fm->fm_extent_count = MAX_EXTENTS;
	if (ioctl(fd, FS_IOC_FIEMAP, fm) < 0) {
		int rc = -errno;
		free(fm);
		return rc;
	}
	uint32_t n = fm->fm_mapped_extents;
	if (n == 0 || !(fm->fm_extents[n - 1].fe_flags & FIEMAP_EXTENT_LAST)) {
		free(fm);
		return -E2BIG;
	}
	for (uint32_t i = 0; i < n; i++) {
		const struct fiemap_extent *fe = &fm->fm_extents[i];
		if (fe->fe_flags &
		    (FIEMAP_EXTENT_UNWRITTEN | FIEMAP_EXTENT_DELALLOC |
		     FIEMAP_EXTENT_ENCODED | FIEMAP_EXTENT_DATA_INLINE)) {
			free(fm);
			return -ENOTSUP;
		}
		out[i].file_off = fe->fe_logical;
		out[i].dev_off = fe->fe_physical;
		out[i].len = fe->fe_length;
		if (out[i].file_off + out[i].len > FILE_SIZE)
			out[i].len = FILE_SIZE - out[i].file_off;
	}
	free(fm);
	*n_out = n;
	return 0;
}

/* Compare n bytes of host memory with the pattern starting at file offset
 * file_off. Returns the number of mismatching bytes. */
static size_t
check_pattern(const unsigned char *host, uint64_t file_off, size_t n)
{
	size_t bad = 0;
	for (size_t i = 0; i < n; i++)
		if (host[i] != pattern_byte((off_t)(file_off + i)))
			bad++;
	return bad;
}

/* Raw read of n bytes at device offset dev_off into gpu, copied to host.
 * Returns the backend's result (bytes or -error). */
static ssize_t
raw_read(opends_handle_t fh, void *gpu, unsigned char *host, size_t n,
         uint64_t dev_off)
{
	cuda_buf_zero(gpu, n);
	ssize_t r = opends_sync_read(fh, gpu, n, (off_t)dev_off, 0);
	if (r > 0)
		cuda_buf_to_host(host, gpu, (size_t)r);
	return r;
}

static ssize_t
raw_write(opends_handle_t fh, void *gpu, const unsigned char *host, size_t n,
          uint64_t dev_off)
{
	cuda_buf_from_host(gpu, host, n);
	return opends_sync_write(fh, gpu, n, (off_t)dev_off, 0);
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

	uint64_t dev_nbytes = 0;
	if (ioctl(raw_fd, BLKGETSIZE64, &dev_nbytes) < 0) {
		fprintf(stderr, "BLKGETSIZE64: %s\n", strerror(errno));
		failed++;
	}

	struct ext ext[MAX_EXTENTS];
	uint32_t n_ext = 0;
	int frc = fiemap_pattern(a.fd, ext, &n_ext);
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

	/* ---- 4. Unaligned raw read (head and tail bounce). ---- */
	{
		size_t n = PAGE - 2;
		ssize_t r = raw_read(raw_fh, gpu, host, n, ext[0].dev_off + 1);
		size_t bad =
		        r == (ssize_t)n
		                ? check_pattern(host, ext[0].file_off + 1, n)
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
		for (size_t i = 0; i < PAGE; i++)
			inv[i] = pattern_byte((off_t)(ext[0].file_off + i));
		w = raw_write(raw_fh, gpu, inv, PAGE, ext[0].dev_off);
		ssize_t r = raw_read(raw_fh, gpu, host, PAGE, ext[0].dev_off);
		if (w != PAGE || r != PAGE ||
		    check_pattern(host, ext[0].file_off, PAGE)) {
			fprintf(stderr,
			        "restore failed (write %zd, read %zd); "
			        "rerun test_sync_read_prep\n",
			        w, r);
			failed++;
		} else {
			fprintf(stderr, "pattern restored ok\n");
		}
		free(inv);
	}

	/* ---- 7. Unaligned raw write is refused, nothing written. ---- */
	{
		cuda_buf_zero(gpu, PAGE);
		ssize_t w = opends_sync_write(raw_fh, gpu, PAGE,
		                              (off_t)(ext[0].dev_off + 1), 0);
		ssize_t r = raw_read(raw_fh, gpu, host, PAGE, ext[0].dev_off);
		if (w != -(ssize_t)OPENDS_INVALID_VALUE || r != PAGE ||
		    check_pattern(host, ext[0].file_off, PAGE)) {
			fprintf(stderr,
			        "unaligned raw write: returned %zd "
			        "(want %d), pattern intact=%s\n",
			        w, -OPENDS_INVALID_VALUE,
			        r == PAGE ? "yes" : "no");
			failed++;
		} else {
			fprintf(stderr, "unaligned raw write refused ok\n");
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
