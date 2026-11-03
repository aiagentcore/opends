/* SPDX-License-Identifier: BSD-3-Clause */
/*
 * bench_raw_block_aisio - Raw block mode against the file-backed path, on
 * the same blocks.
 *
 * Takes a file on the mounted XFS and the ublk device that file system sits
 * on, registers both, and drives the file's largest plain extent through
 * each handle: the file handle resolves offsets through the extent index,
 * the raw handle addresses the namespace directly. Reads go GPU-bound P2P
 * either way. Writes through the file handle take the kernel path (GPU to
 * host bounce, then pwrite on the O_DIRECT fd); through the raw handle they
 * are P2P DMA. The write legs overwrite the file's blocks, so the harness
 * deletes the file afterwards.
 *
 * For each handle, I/O family (sync, async, stream, batch) and (chunk, qd)
 * pair of the sweep, the region is read or written once, chunk by chunk
 * with qd operations in flight, and the wall-clock throughput is printed as
 * one RESULT line on stdout. Before the read legs and after the write legs
 * the first chunk is read through both handles and compared, so a result
 * is only reported for two handles that address the same blocks. Stream
 * legs drain their streams every STREAM_WINDOW ops per stream (see the
 * stream case for why).
 *
 * usage: bench_raw_block_aisio <raw_block_dev> <file> [region_bytes]
 *            [chunk_list] [qd_list]
 *   chunk_list and qd_list are comma-separated. Defaults: 2 GiB,
 *   65536,262144,1048576,4194304,16777216 and 1,8,32. sync runs at qd 1
 *   only. The GPU buffer is max(chunk) * max(qd) bytes, registered with
 *   opends_buf_register. BENCH_RAW_FIRST=1 runs the raw handle first.
 */
#define _GNU_SOURCE

#include "opends.h"
#include "test_cuda_common.h"

#include <cuda.h>

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <linux/fiemap.h>
#include <linux/fs.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#define ENV_RAW_BLOCK "OPENDS_AISIO_RAW_BLOCK"
#define MAX_EXTENTS 1024
#define MAX_LIST 16
#define MAX_STREAMS 64
#define CHECK_BYTES ((size_t)1 << 20)
#define STREAM_WINDOW 64 /* stream ops queued per stream before a drain */

struct region {
	uint64_t file_off;
	uint64_t dev_off;
	uint64_t len;
};

struct handles {
	opends_handle_t file;
	opends_handle_t raw;
};

enum mode { MODE_SYNC, MODE_ASYNC, MODE_STREAM, MODE_BATCH, MODE_COUNT };
static const char *const mode_name[MODE_COUNT] = {"sync", "async", "stream",
                                                  "batch"};

static CUstream streams[MAX_STREAMS];
static int n_streams;

static double
now_s(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int
sync_streams(unsigned n)
{
	int rc = 0;
	for (unsigned s = 0; s < n; s++) {
		if (cuStreamSynchronize(streams[s]) != CUDA_SUCCESS) {
			fprintf(stderr, "  cuStreamSynchronize(%u)\n", s);
			rc = -1;
		}
	}
	return rc;
}

/* The largest extent of fd that is plain written data. */
static int
largest_extent(int fd, struct region *out)
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
	out->len = 0;
	for (uint32_t i = 0; i < n; i++) {
		const struct fiemap_extent *fe = &fm->fm_extents[i];
		if (fe->fe_flags &
		    (FIEMAP_EXTENT_UNWRITTEN | FIEMAP_EXTENT_DELALLOC |
		     FIEMAP_EXTENT_ENCODED | FIEMAP_EXTENT_DATA_INLINE))
			continue;
		if (fe->fe_length > out->len) {
			out->file_off = fe->fe_logical;
			out->dev_off = fe->fe_physical;
			out->len = fe->fe_length;
		}
	}
	free(fm);
	if (out->len == 0)
		return -ENOENT;
	fprintf(stderr,
	        "file has %u extent(s); largest %" PRIu64 " bytes at "
	        "file offset %" PRIu64 ", device offset %" PRIu64 "\n",
	        n, out->len, out->file_off, out->dev_off);
	return 0;
}

static int
parse_list(const char *s, uint64_t *out, int max)
{
	int n = 0;
	char *dup = strdup(s);
	for (char *tok = strtok(dup, ","); tok && n < max;
	     tok = strtok(NULL, ",")) {
		out[n] = strtoull(tok, NULL, 0);
		if (out[n] == 0) {
			free(dup);
			return -1;
		}
		n++;
	}
	free(dup);
	return n;
}

/*
 * One leg: nchunks operations of chunk bytes, sequential over the region
 * starting at off (a file offset or a device offset, the handle decides),
 * qd in flight. Returns 0 and the elapsed seconds, or -1 after a message.
 */
static int
run_leg(opends_handle_t fh, enum mode mode, bool write, uint8_t *buf,
        uint64_t off, size_t chunk, unsigned nchunks, unsigned qd, double *secs)
{
	double t0 = now_s();
	int rc = 0;

	switch (mode) {
	case MODE_SYNC:
		for (unsigned i = 0; i < nchunks && rc == 0; i++) {
			off_t fo = (off_t)(off + (uint64_t)i * chunk);
			ssize_t n =
			        write ? opends_sync_write(fh, buf, chunk, fo, 0)
			              : opends_sync_read(fh, buf, chunk, fo, 0);
			if (n != (ssize_t)chunk) {
				fprintf(stderr, "  sync op %u: %zd\n", i, n);
				rc = -1;
			}
		}
		break;

	case MODE_ASYNC: {
		opends_async_future_t *fut = calloc(qd, sizeof(*fut));
		for (unsigned i = 0; i < nchunks && rc == 0; i++) {
			unsigned slot = i % qd;
			if (i >= qd) {
				ssize_t n = opends_async_await(&fut[slot]);
				if (n != (ssize_t)chunk) {
					fprintf(stderr, "  async op %u: %zd\n",
					        i - qd, n);
					rc = -1;
					break;
				}
			}
			off_t fo = (off_t)(off + (uint64_t)i * chunk);
			off_t bo = (off_t)(slot * chunk);
			opends_error_t err =
			        write ? opends_async_write(fh, buf, chunk, fo,
			                                   bo, &fut[slot])
			              : opends_async_read(fh, buf, chunk, fo,
			                                  bo, &fut[slot]);
			if (err.err != OPENDS_SUCCESS) {
				fprintf(stderr, "  async submit %u: %s\n", i,
				        opends_op_status_error(err.err));
				rc = -1;
			}
		}
		unsigned tail = nchunks < qd ? nchunks : qd;
		for (unsigned i = nchunks - tail; i < nchunks; i++) {
			ssize_t n = opends_async_await(&fut[i % qd]);
			if (n != (ssize_t)chunk && rc == 0) {
				fprintf(stderr, "  async op %u: %zd\n", i, n);
				rc = -1;
			}
		}
		free(fut);
		break;
	}

	case MODE_STREAM: {
		/* Stream ops read their parameters at execution time, so each
		 * op keeps its own until its stream has drained. qd streams,
		 * round robin; ops on one stream complete in order and reuse
		 * that stream's slot of the buffer. At most STREAM_WINDOW ops
		 * per stream are queued before the streams are drained: a file
		 * handle's stream write bounces through host memory with a CUDA
		 * copy on the I/O worker, and once enough ops fill the CUDA
		 * command queue the submitting thread blocks inside CUDA
		 * holding the lock that copy needs, while the queued work waits
		 * on the worker. The bound keeps both handles on the same
		 * footing. */
		size_t *sz = calloc(nchunks, sizeof(*sz));
		off_t *fo = calloc(nchunks, sizeof(*fo));
		off_t *bo = calloc(nchunks, sizeof(*bo));
		ssize_t *bytes = calloc(nchunks, sizeof(*bytes));
		unsigned window = qd * STREAM_WINDOW;
		for (unsigned i = 0; i < nchunks && rc == 0; i++) {
			unsigned s = i % qd;
			sz[i] = chunk;
			fo[i] = (off_t)(off + (uint64_t)i * chunk);
			bo[i] = (off_t)(s * chunk);
			bytes[i] = -1;
			opends_error_t err =
			        write ? opends_stream_write(
			                        fh, buf, &sz[i], &fo[i], &bo[i],
			                        &bytes[i], streams[s])
			              : opends_stream_read(
			                        fh, buf, &sz[i], &fo[i], &bo[i],
			                        &bytes[i], streams[s]);
			if (err.err != OPENDS_SUCCESS) {
				fprintf(stderr, "  stream submit %u: %s\n", i,
				        opends_op_status_error(err.err));
				rc = -1;
			}
			if ((i + 1) % window == 0 && sync_streams(qd) < 0)
				rc = -1;
		}
		if (sync_streams(qd) < 0)
			rc = -1;
		for (unsigned i = 0; i < nchunks && rc == 0; i++) {
			if (bytes[i] != (ssize_t)chunk) {
				fprintf(stderr, "  stream op %u: %zd\n", i,
				        bytes[i]);
				rc = -1;
			}
		}
		free(sz);
		free(fo);
		free(bo);
		free(bytes);
		break;
	}

	case MODE_BATCH: {
		/* qd ops per submit, wait for all of them, repeat. */
		opends_batch_handle_t batch = NULL;
		opends_error_t err = opends_batch_setup(&batch, qd);
		if (err.err != OPENDS_SUCCESS) {
			fprintf(stderr, "  batch_setup: %s\n",
			        opends_op_status_error(err.err));
			return -1;
		}
		opends_io_params_t *io = calloc(qd, sizeof(*io));
		opends_io_events_t *ev = calloc(qd, sizeof(*ev));
		for (unsigned i = 0; i < nchunks && rc == 0; i += qd) {
			unsigned n = nchunks - i < qd ? nchunks - i : qd;
			for (unsigned j = 0; j < n; j++) {
				memset(&io[j], 0, sizeof(io[j]));
				io[j].mode = OPENDS_BATCH;
				io[j].fh = fh;
				io[j].opcode =
				        write ? OPENDS_WRITE : OPENDS_READ;
				io[j].cookie = &io[j];
				io[j].u.batch.dev_ptr_base = buf;
				io[j].u.batch.dev_ptr_offset =
				        (off_t)(j * chunk);
				io[j].u.batch.file_offset =
				        (off_t)(off +
				                (uint64_t)(i + j) * chunk);
				io[j].u.batch.size = chunk;
			}
			err = opends_batch_submit(batch, n, io, 0);
			if (err.err != OPENDS_SUCCESS) {
				fprintf(stderr, "  batch submit %u: %s\n", i,
				        opends_op_status_error(err.err));
				rc = -1;
				break;
			}
			unsigned nr = n;
			err = opends_batch_get_status(batch, n, &nr, ev, NULL);
			if (err.err != OPENDS_SUCCESS || nr != n) {
				fprintf(stderr,
				        "  batch get_status %u: %s, %u of %u\n",
				        i, opends_op_status_error(err.err), nr,
				        n);
				rc = -1;
				break;
			}
			for (unsigned j = 0; j < n; j++) {
				if (ev[j].status != OPENDS_COMPLETE ||
				    ev[j].ret != chunk) {
					fprintf(stderr,
					        "  batch op %u: status %d, "
					        "ret %zu\n",
					        i + j, ev[j].status, ev[j].ret);
					rc = -1;
				}
			}
		}
		free(io);
		free(ev);
		opends_batch_destroy(batch);
		break;
	}

	default: return -1;
	}

	*secs = now_s() - t0;
	return rc;
}

/* Read the first CHECK_BYTES of the region through both handles and compare
 * the bytes. Returns 0 when they agree. */
static int
check_same_blocks(const struct handles *h, const struct region *r, uint8_t *buf,
                  const char *when)
{
	size_t n = r->len < CHECK_BYTES ? (size_t)r->len : CHECK_BYTES;
	unsigned char *a = malloc(n);
	unsigned char *b = malloc(n);
	int rc = -1;
	if (!a || !b)
		goto out;
	cuda_buf_zero(buf, n);
	ssize_t ra = opends_sync_read(h->file, buf, n, (off_t)r->file_off, 0);
	cuda_buf_to_host(a, buf, n);
	cuda_buf_zero(buf, n);
	ssize_t rb = opends_sync_read(h->raw, buf, n, (off_t)r->dev_off, 0);
	cuda_buf_to_host(b, buf, n);
	if (ra != (ssize_t)n || rb != (ssize_t)n || memcmp(a, b, n) != 0) {
		fprintf(stderr,
		        "oracle %s: file read %zd, raw read %zd, bytes %s\n",
		        when, ra, rb,
		        ra == rb && ra == (ssize_t)n ? "differ" : "n/a");
		goto out;
	}
	fprintf(stderr, "oracle %s: file and raw handles read the same bytes\n",
	        when);
	rc = 0;
out:
	free(a);
	free(b);
	return rc;
}

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
	if (argc < 3 || argc > 6) {
		fprintf(stderr,
		        "usage: %s <raw_block_dev> <file> [region_bytes] "
		        "[chunk_list] [qd_list]\n",
		        argv[0]);
		return 1;
	}
	const char *raw_path = argv[1];
	const char *file_path = argv[2];
	uint64_t region_bytes =
	        argc > 3 ? strtoull(argv[3], NULL, 0) : (uint64_t)2 << 30;
	uint64_t chunks[MAX_LIST], qds[MAX_LIST];
	int n_chunks = parse_list(
	        argc > 4 ? argv[4] : "65536,262144,1048576,4194304,16777216",
	        chunks, MAX_LIST);
	int n_qds = parse_list(argc > 5 ? argv[5] : "1,8,32", qds, MAX_LIST);
	if (n_chunks <= 0 || n_qds <= 0 || region_bytes == 0) {
		fprintf(stderr, "bad sweep lists or region\n");
		return 1;
	}
	size_t max_chunk = 0, max_qd = 0;
	for (int i = 0; i < n_chunks; i++)
		if (chunks[i] > max_chunk)
			max_chunk = chunks[i];
	for (int i = 0; i < n_qds; i++)
		if (qds[i] > max_qd)
			max_qd = qds[i];
	if (max_qd > MAX_STREAMS) {
		fprintf(stderr, "qd above %d not supported\n", MAX_STREAMS);
		return 1;
	}

	CUcontext cuctx;
	if (cuda_ctx_create(&cuctx) < 0) {
		fprintf(stderr, "cuCtxCreate failed\n");
		return 1;
	}
	setenv(ENV_RAW_BLOCK, "1", 1);
	opends_error_t err = opends_driver_open();
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "driver_open: %s\n",
		        opends_op_status_error(err.err));
		return 1;
	}

	int file_fd = open(file_path, O_RDWR | O_DIRECT);
	int raw_fd = open(raw_path, O_RDWR | O_DIRECT);
	if (file_fd < 0 || raw_fd < 0) {
		fprintf(stderr, "open: %s\n", strerror(errno));
		return 1;
	}
	struct handles h = {NULL, NULL};
	err = opends_handle_register(&h.file, file_fd);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "register %s: %s\n", file_path,
		        opends_op_status_error(err.err));
		return 1;
	}
	err = opends_handle_register(&h.raw, raw_fd);
	if (err.err != OPENDS_SUCCESS) {
		fprintf(stderr, "register %s: %s\n", raw_path,
		        opends_op_status_error(err.err));
		return 1;
	}

	struct region r;
	int frc = largest_extent(file_fd, &r);
	if (frc < 0) {
		fprintf(stderr, "FIEMAP(%s): %s\n", file_path, strerror(-frc));
		return 1;
	}
	if (r.len > region_bytes)
		r.len = region_bytes;
	r.len -= r.len % max_chunk;
	if (r.len == 0) {
		fprintf(stderr, "region smaller than the largest chunk\n");
		return 1;
	}
	fprintf(stderr, "region: %" PRIu64 " bytes\n", r.len);

	size_t buf_bytes = max_chunk * max_qd;
	uint8_t *buf = cuda_register_acquire(buf_bytes);
	if (!buf) {
		fprintf(stderr, "GPU buffer of %zu bytes failed\n", buf_bytes);
		return 1;
	}
	for (n_streams = 0; n_streams < (int)max_qd; n_streams++) {
		if (cuStreamCreate(&streams[n_streams],
		                   CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS ||
		    opends_stream_register(streams[n_streams], 0).err !=
		            OPENDS_SUCCESS) {
			fprintf(stderr, "stream %d setup failed\n", n_streams);
			return 1;
		}
	}

	/* BENCH_RAW_FIRST=1 runs the raw handle's legs before the file's, to
	 * tell a handle effect from an order effect. */
	const char *rf = getenv("BENCH_RAW_FIRST");
	bool raw_first = rf && rf[0] == '1';
	int failed = 0;
	if (check_same_blocks(&h, &r, buf, "before") < 0)
		failed++;

	printf("# region_bytes=%" PRIu64 " file_off=%" PRIu64
	       " dev_off=%" PRIu64 " buf_bytes=%zu\n",
	       r.len, r.file_off, r.dev_off, buf_bytes);
	for (int write = 0; write < 2; write++) {
		for (int hi = 0; hi < 2; hi++) {
			bool is_raw = raw_first ? !hi : hi;
			opends_handle_t fh = is_raw ? h.raw : h.file;
			const char *hname = is_raw ? "raw" : "file";
			uint64_t off = is_raw ? r.dev_off : r.file_off;
			for (int m = 0; m < MODE_COUNT; m++) {
				for (int ci = 0; ci < n_chunks; ci++) {
					size_t chunk = chunks[ci];
					unsigned nchunks = r.len / chunk;
					for (int qi = 0; qi < n_qds; qi++) {
						unsigned qd = qds[qi];
						if (m == MODE_SYNC && qd != 1)
							continue;
						double secs = 0;
						int rc = run_leg(fh, m, write,
						                 buf, off,
						                 chunk, nchunks,
						                 qd, &secs);
						if (rc < 0) {
							printf("RESULT "
							       "handle=%s "
							       "op=%s mode=%s "
							       "chunk=%zu "
							       "qd=%u "
							       "error=1\n",
							       hname,
							       write ? "write"
							             : "read",
							       mode_name[m],
							       chunk, qd);
							failed++;
							continue;
						}
						double mibs = (double)r.len /
						              (1 << 20) / secs;
						printf("RESULT handle=%s op=%s "
						       "mode=%s chunk=%zu "
						       "qd=%u bytes=%" PRIu64
						       " secs=%.4f "
						       "MiBs=%.1f\n",
						       hname,
						       write ? "write" : "read",
						       mode_name[m], chunk, qd,
						       r.len, secs, mibs);
						fflush(stdout);
					}
				}
			}
		}
	}

	if (check_same_blocks(&h, &r, buf, "after") < 0)
		failed++;

	for (int i = 0; i < n_streams; i++) {
		opends_stream_deregister(streams[i]);
		cuStreamDestroy(streams[i]);
	}
	cuda_register_release(buf);
	opends_handle_deregister(h.raw);
	opends_handle_deregister(h.file);
	close(raw_fd);
	close(file_fd);
	opends_driver_close();
	cuCtxDestroy(cuctx);

	if (failed)
		fprintf(stderr, "%d leg(s) failed\n", failed);
	else
		fprintf(stderr, "all legs ok\n");
	return failed ? 1 : 0;
}
