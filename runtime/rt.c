// Lumen actor runtime: single-threaded event loop with message queue.
// Compiled to rt.o and linked with Lumen programs.

#ifdef __APPLE__
#define _XOPEN_SOURCE 600
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <signal.h>
#include <math.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ucontext.h>
#ifdef __linux__
#include <sys/epoll.h>
#endif
#ifdef __APPLE__
#include <sys/event.h>
#endif
#include <fcntl.h>
#include <errno.h>

// Forward declarations.
void lumen_rt_drain(void);
int lumen_rt_step(void);

// --- Allocator dispatch --------------------------------------------------
//
// MVP has two allocators: the default RC-malloc (per-allocation malloc
// + 8-byte header carrying rc + 0x4C554D45 sentinel) and the arena
// allocator (bump-allocate from a region; whole region freed at once,
// zeroed header so rc_incr/decr skip arena memory). rc_incr/rc_decr
// already no-op on memory without the sentinel — arena allocations are
// invisible to RC by construction.
//
// The pointer-indirection via LumenAllocator is deliberately
// generalized: later tickets can graduate this into a pluggable vtable
// for user allocators without touching every call site.

typedef struct {
    // Return a payload pointer. The block has 8 bytes of header space
    // BEFORE the returned pointer (header lives at ptr-8). For the
    // default allocator that header carries rc+magic; for arena it's
    // zeroed so the sentinel check skips it.
    void *(*alloc)(void *state, int64_t size);
    void *state;
} LumenAllocator;

static void *rc_malloc_alloc(void *state, int64_t size) {
    (void)state;
    char *raw = (char *)malloc(8 + (size_t)size);
    if (!raw) { fprintf(stderr, "FATAL: rc_alloc oom (size=%lld)\n", (long long)size); abort(); }
    *(int32_t *)(raw + 0) = 1;            // rc = 1
    *(int32_t *)(raw + 4) = 0x4C554D45;   // "LUME" sentinel
    return raw + 8;
}

static LumenAllocator rc_malloc_allocator = { rc_malloc_alloc, NULL };
// Per-thread allocator stack — each OS thread carries its own active
// allocator so arena blocks on one worker don't affect another's
// RC-malloc path. __thread with a constant-address initializer is
// POSIX + ELF-TLS compatible.
static __thread LumenAllocator *current_allocator = &rc_malloc_allocator;

// --- Arena allocator -----------------------------------------------------

typedef struct {
    char *base;
    char *cur;
    char *end;
} LumenArena;

static void *arena_alloc(void *state, int64_t size) {
    LumenArena *a = (LumenArena *)state;
    int64_t need = 8 + size;
    if (a->cur + need > a->end) {
        int64_t used = (int64_t)(a->cur - a->base);
        int64_t cap = (int64_t)(a->end - a->base);
        int64_t new_cap = cap * 2;
        if (new_cap < used + need) new_cap = used + need + 4096;
        char *new_base = (char *)realloc(a->base, (size_t)new_cap);
        if (!new_base) { fprintf(stderr, "FATAL: arena grow oom\n"); abort(); }
        a->base = new_base;
        a->cur = new_base + used;
        a->end = new_base + new_cap;
    }
    char *raw = a->cur;
    a->cur += need;
    // Zero the header so the sentinel check in rc_incr/rc_decr fails.
    *(int32_t *)(raw + 0) = 0;
    *(int32_t *)(raw + 4) = 0;
    return raw + 8;
}

int64_t lumen_arena_new(int64_t initial_size) {
    LumenArena *a = (LumenArena *)malloc(sizeof(LumenArena));
    if (!a) { fprintf(stderr, "FATAL: arena_new oom\n"); abort(); }
    if (initial_size < 4096) initial_size = 4096;
    a->base = (char *)malloc((size_t)initial_size);
    if (!a->base) { fprintf(stderr, "FATAL: arena_new region oom\n"); abort(); }
    a->cur = a->base;
    a->end = a->base + initial_size;
    return (int64_t)(uintptr_t)a;
}

void lumen_arena_free(int64_t arena) {
    LumenArena *a = (LumenArena *)(uintptr_t)arena;
    if (!a) return;
    free(a->base);
    free(a);
}

// Swap the current allocator to one backed by `arena`, returning the
// previous allocator pointer so the emitted block can restore it on
// exit. The slot struct is malloc'd so nested arenas can each carry
// distinct state pointers.
int64_t lumen_allocator_push_arena(int64_t arena) {
    LumenArena *a = (LumenArena *)(uintptr_t)arena;
    LumenAllocator *prev = current_allocator;
    LumenAllocator *slot = (LumenAllocator *)malloc(sizeof(LumenAllocator));
    if (!slot) { fprintf(stderr, "FATAL: allocator slot oom\n"); abort(); }
    slot->alloc = arena_alloc;
    slot->state = a;
    current_allocator = slot;
    return (int64_t)(uintptr_t)prev;
}

void lumen_allocator_pop(int64_t prev_allocator) {
    LumenAllocator *prev = (LumenAllocator *)(uintptr_t)prev_allocator;
    if (current_allocator != &rc_malloc_allocator) {
        free(current_allocator);
    }
    current_allocator = prev;
}

// Core rc_alloc: dispatches through current_allocator. Replaces the
// cranelift-defined helper of the same name.
int64_t lumen_rc_alloc(int64_t size) {
    void *payload = current_allocator->alloc(current_allocator->state, size);
    return (int64_t)(uintptr_t)payload;
}

// Resize a payload: for the default allocator this is a real libc
// realloc (preserves the rc header). For the arena, we allocate a
// fresh block and memcpy — the old block stays in the arena until
// reset/free. Used by list/map grow paths so the whole data structure
// lives in whatever allocator was active when it was created.
static void *lumen_realloc_raw(void *old_payload, int64_t old_size, int64_t new_size) {
    if (current_allocator == &rc_malloc_allocator) {
        char *old_raw = (char *)old_payload - 8;
        char *new_raw = (char *)realloc(old_raw, 8 + (size_t)new_size);
        if (!new_raw) { fprintf(stderr, "FATAL: realloc oom (size=%lld)\n", (long long)new_size); abort(); }
        return new_raw + 8;
    }
    void *new_payload = current_allocator->alloc(current_allocator->state, new_size);
    int64_t copy = old_size < new_size ? old_size : new_size;
    if (copy > 0) memcpy(new_payload, old_payload, (size_t)copy);
    return new_payload;
}

// --- Worker thread pool + actor scheduler --------------------------------
//
// Worker pool is live (lumen-u53) but actor dispatch stays on the
// caller's thread — either the main thread's cooperative scheduler
// or an inline drain in lumen_rt_ask. The per-bucket-mutex approach
// from the earlier cv1 attempt gave mutual exclusion but NOT FIFO
// per actor: POSIX mutexes don't preserve wait order, so multiple
// workers contending for the same bucket can dispatch messages out
// of send order.
//
// Correct multi-worker scheduling needs per-actor state (mailbox +
// sequence counter). Tracked as lumen-cv1 (reopened).

#include <pthread.h>

#define LUMEN_MAX_WORKERS 64
static pthread_t lumen_workers[LUMEN_MAX_WORKERS];
static int32_t lumen_worker_count = 0;
static pthread_mutex_t lumen_worker_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t lumen_worker_wake = PTHREAD_COND_INITIALIZER;
static volatile int32_t lumen_worker_shutdown = 0;

static int32_t lumen_worker_count_from_env(void) {
    const char *env = getenv("LUMEN_THREADS");
    if (env) {
        int n = atoi(env);
        if (n >= 0 && n <= LUMEN_MAX_WORKERS) return n;
    }
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n <= 0) n = 4;
    if (n > LUMEN_MAX_WORKERS) n = LUMEN_MAX_WORKERS;
    return (int32_t)n;
}

static void *lumen_worker_main(void *arg) {
    (void)arg;
    pthread_mutex_lock(&lumen_worker_mu);
    while (!lumen_worker_shutdown) {
        pthread_cond_wait(&lumen_worker_wake, &lumen_worker_mu);
    }
    pthread_mutex_unlock(&lumen_worker_mu);
    return NULL;
}

__attribute__((constructor))
static void lumen_worker_pool_start(void) {
    lumen_worker_count = lumen_worker_count_from_env();
    for (int i = 0; i < lumen_worker_count; i++) {
        if (pthread_create(&lumen_workers[i], NULL, lumen_worker_main, NULL) != 0) {
            lumen_worker_count = i;
            break;
        }
    }
}

__attribute__((destructor))
static void lumen_worker_pool_stop(void) {
    pthread_mutex_lock(&lumen_worker_mu);
    lumen_worker_shutdown = 1;
    pthread_cond_broadcast(&lumen_worker_wake);
    pthread_mutex_unlock(&lumen_worker_mu);
    for (int i = 0; i < lumen_worker_count; i++) {
        pthread_join(lumen_workers[i], NULL);
    }
}

// --- Message queue -------------------------------------------------------

typedef struct {
    void *target_cell;
    void (*dispatch)(void *cell, int32_t kind, int64_t arg0, int64_t *reply);
    int32_t msg_kind;
    int64_t arg0;
    int64_t *reply_slot;
} QueueEntry;

#define QUEUE_CAP 65536
static QueueEntry queue[QUEUE_CAP];
static int queue_head = 0;
static int queue_tail = 0;

void lumen_rt_send(void *cell,
                   void (*dispatch)(void *, int32_t, int64_t, int64_t *),
                   int32_t kind, int64_t arg0) {
    int next = (queue_tail + 1) % QUEUE_CAP;
    if (next == queue_head) {
        fprintf(stderr, "FATAL: actor message queue full (%d messages)\n", QUEUE_CAP);
        return;
    }
    queue[queue_tail].target_cell = cell;
    queue[queue_tail].dispatch = dispatch;
    queue[queue_tail].msg_kind = kind;
    queue[queue_tail].arg0 = arg0;
    queue[queue_tail].reply_slot = NULL;
    queue_tail = next;
}

int64_t lumen_rt_ask(void *cell,
                     void (*dispatch)(void *, int32_t, int64_t, int64_t *),
                     int32_t kind, int64_t arg0) {
    int64_t reply = 0;
    int next = (queue_tail + 1) % QUEUE_CAP;
    if (next == queue_head) {
        fprintf(stderr, "FATAL: actor message queue full (%d messages)\n", QUEUE_CAP);
        return 0;
    }
    queue[queue_tail].target_cell = cell;
    queue[queue_tail].dispatch = dispatch;
    queue[queue_tail].msg_kind = kind;
    queue[queue_tail].arg0 = arg0;
    queue[queue_tail].reply_slot = &reply;
    queue_tail = next;
    lumen_rt_drain();
    return reply;
}

int lumen_rt_step(void) {
    if (queue_head == queue_tail) return 0;
    QueueEntry *e = &queue[queue_head];
    queue_head = (queue_head + 1) % QUEUE_CAP;
    int64_t reply = 0;
    e->dispatch(e->target_cell, e->msg_kind, e->arg0, &reply);
    if (e->reply_slot) {
        *e->reply_slot = reply;
    }
    return 1;
}

void lumen_rt_drain(void) {
    while (lumen_rt_step()) {}
}

void lumen_rt_yield(void) {
    lumen_rt_step();
}

// --- TCP socket operations ------------------------------------------------

// Allocate a Lumen bytes value: [rc:i32=1 | magic:i32=0x4C554D45 | len:i32 | data...]
// Returns pointer to the payload (len field). Routes through the
// current allocator so bytes created inside an arena live in the arena.
static char *alloc_bytes(int32_t len) {
    char *payload = (char *)current_allocator->alloc(
        current_allocator->state, (int64_t)(4 + len));
    *(int32_t *)(payload) = len;
    return payload;
}

// Bytes length.
int32_t lumen_bytes_len(int64_t bytes_ptr) {
    if (bytes_ptr == 0) return 0;
    return *(int32_t *)(uintptr_t)bytes_ptr;
}

// Bytes get: read one byte at index, zero-extend to i32.
int32_t lumen_bytes_get(int64_t bytes_ptr, int32_t index) {
    char *buf = (char *)(uintptr_t)bytes_ptr;
    return (int32_t)(unsigned char)buf[4 + index];
}

// Encode a Unicode scalar value into UTF-8 bytes. `out` must have room
// for at least 4 bytes. Returns the number of bytes written (1..=4).
// Used by codegen for char formatting and by the std/char module.
int32_t lumen_utf8_encode(int32_t cp_i32, char *out) {
    uint32_t cp = (uint32_t)cp_i32;
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

// Decode one UTF-8 scalar starting at byte offset `byte_i` in `bytes_ptr`.
// Returns the scalar value, or -1 if the byte position is invalid
// (continuation byte, truncated sequence, bad leading byte).
int32_t lumen_utf8_decode_at(int64_t bytes_ptr, int32_t byte_i) {
    char *buf = (char *)(uintptr_t)bytes_ptr;
    int32_t len = *(int32_t *)buf;
    if (byte_i < 0 || byte_i >= len) return -1;
    unsigned char *p = (unsigned char *)(buf + 4 + byte_i);
    unsigned char b0 = p[0];
    if (b0 < 0x80) return (int32_t)b0;
    if ((b0 & 0xE0) == 0xC0) {
        if (byte_i + 1 >= len) return -1;
        return ((b0 & 0x1F) << 6) | (p[1] & 0x3F);
    }
    if ((b0 & 0xF0) == 0xE0) {
        if (byte_i + 2 >= len) return -1;
        return ((b0 & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
    }
    if ((b0 & 0xF8) == 0xF0) {
        if (byte_i + 3 >= len) return -1;
        return ((b0 & 0x07) << 18) | ((p[1] & 0x3F) << 12)
             | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
    }
    return -1;
}

// How many UTF-8 bytes does the scalar at `byte_i` occupy? 1..=4, or 0
// on invalid position. Used to advance a byte-based iterator one char
// forward.
int32_t lumen_utf8_char_width(int64_t bytes_ptr, int32_t byte_i) {
    char *buf = (char *)(uintptr_t)bytes_ptr;
    int32_t len = *(int32_t *)buf;
    if (byte_i < 0 || byte_i >= len) return 0;
    unsigned char b0 = (unsigned char)buf[4 + byte_i];
    if (b0 < 0x80) return 1;
    if ((b0 & 0xE0) == 0xC0) return 2;
    if ((b0 & 0xF0) == 0xE0) return 3;
    if ((b0 & 0xF8) == 0xF0) return 4;
    return 0;
}

// Count scalar values in a UTF-8 byte buffer. O(n).
int32_t lumen_string_char_count(int64_t bytes_ptr) {
    char *buf = (char *)(uintptr_t)bytes_ptr;
    int32_t len = *(int32_t *)buf;
    int32_t count = 0;
    int32_t i = 0;
    while (i < len) {
        int32_t w = lumen_utf8_char_width(bytes_ptr, i);
        if (w == 0) return count; // bail on malformed tail
        count++;
        i += w;
    }
    return count;
}

// Walk the UTF-8 buffer returning the scalar at char index `char_i`,
// or -1 if out of range. Matches the user expectation that
// string.char_at(s, 2) gives the third character, not the third byte.
int32_t lumen_string_char_at(int64_t bytes_ptr, int32_t char_i) {
    if (char_i < 0) return -1;
    char *buf = (char *)(uintptr_t)bytes_ptr;
    int32_t len = *(int32_t *)buf;
    int32_t byte_i = 0;
    int32_t k = 0;
    while (byte_i < len) {
        if (k == char_i) return lumen_utf8_decode_at(bytes_ptr, byte_i);
        int32_t w = lumen_utf8_char_width(bytes_ptr, byte_i);
        if (w == 0) return -1;
        byte_i += w;
        k++;
    }
    return -1;
}

// Allocate a Lumen string containing the UTF-8 encoding of a single
// scalar value. Used by string.from_char.
int64_t lumen_string_from_char(int32_t cp) {
    char tmp[4];
    int32_t n = lumen_utf8_encode(cp, tmp);
    char *payload = alloc_bytes(n);
    memcpy(payload + 4, tmp, n);
    return (int64_t)(uintptr_t)payload;
}

// Allocate a new zero-filled bytes buffer.
int64_t lumen_bytes_new(int32_t size) {
    char *payload = alloc_bytes(size);
    if (size > 0) memset(payload + 4, 0, size);
    return (int64_t)(uintptr_t)payload;
}

// Identity: bytes and strings have the same representation.
int64_t lumen_bytes_from_string(int64_t s) { return s; }
int64_t lumen_string_from_bytes(int64_t b) { return b; }

// Extract a slice of bytes: returns new bytes [start..start+slice_len].
int64_t lumen_bytes_slice(int64_t bytes_ptr, int32_t start, int32_t slice_len) {
    char *src = (char *)(uintptr_t)bytes_ptr;
    int32_t src_len = *(int32_t *)src;
    if (start < 0) start = 0;
    if (start > src_len) start = src_len;
    if (slice_len < 0) slice_len = 0;
    if (start + slice_len > src_len) slice_len = src_len - start;
    char *payload = alloc_bytes(slice_len);
    if (slice_len > 0) memcpy(payload + 4, src + 4 + start, slice_len);
    return (int64_t)(uintptr_t)payload;
}

// socket + setsockopt + bind + listen. Returns server fd (or -1 on error).
int lumen_tcp_listen(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 128) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

// Accept a connection. Returns client fd (or -1 on error).
int lumen_tcp_accept(int server_fd) {
    struct sockaddr_in client_addr;
    socklen_t len = sizeof(client_addr);
    return accept(server_fd, (struct sockaddr *)&client_addr, &len);
}

// Read up to max bytes from fd. Returns a Lumen bytes pointer (i64).
int64_t lumen_tcp_read(int fd, int max) {
    char *buf = alloc_bytes(max);
    if (!buf) return 0;
    ssize_t n = read(fd, buf + 4, max);  // data starts at offset 4
    if (n < 0) n = 0;
    *(int32_t *)buf = (int32_t)n;         // store actual length
    return (int64_t)(uintptr_t)buf;
}

// Write bytes to fd. bytes_ptr is a Lumen bytes pointer [len:i32 | data...].
// Returns number of bytes written.
int64_t lumen_tcp_write(int fd, int64_t bytes_ptr) {
    char *buf = (char *)(uintptr_t)bytes_ptr;
    int32_t len = *(int32_t *)buf;
    ssize_t n = write(fd, buf + 4, len);
    return (int64_t)n;
}

// Close a file descriptor.
void lumen_tcp_close(int fd) {
    close(fd);
}

// --- HTTP/1.1 parsing and formatting ----------------------------------------

// Helper: create an rc_alloc'd Lumen string from raw C data.
// Uses the same alloc_bytes layout: [rc:i32=1 | magic:0x4C554D45 | len:i32 | data...]
// Returns pointer to [len:i32 | data...] (past the 8-byte rc header).
static int64_t make_lumen_string(const char *data, int len) {
    char *payload = alloc_bytes(len);
    if (len > 0) memcpy(payload + 4, data, len);
    return (int64_t)(uintptr_t)payload;
}

// Parse HTTP method from raw request bytes.
// raw_bytes_ptr points to [len:i32 | data...] (Lumen bytes layout).
int64_t lumen_http_parse_method(int64_t raw_bytes_ptr) {
    char *buf = (char *)(uintptr_t)raw_bytes_ptr;
    int32_t len = *(int32_t *)buf;
    const char *data = buf + 4;

    // Find first space -> method ends there.
    int i = 0;
    while (i < len && data[i] != ' ') i++;

    return make_lumen_string(data, i);
}

// Parse HTTP path from raw request bytes.
int64_t lumen_http_parse_path(int64_t raw_bytes_ptr) {
    char *buf = (char *)(uintptr_t)raw_bytes_ptr;
    int32_t len = *(int32_t *)buf;
    const char *data = buf + 4;

    // Find first space (end of method).
    int i = 0;
    while (i < len && data[i] != ' ') i++;
    i++; // skip the space

    // Path starts here. Find second space.
    int path_start = i;
    while (i < len && data[i] != ' ') i++;

    return make_lumen_string(data + path_start, i - path_start);
}

// Parse HTTP body from raw request bytes.
// Body starts after "\r\n\r\n".
int64_t lumen_http_parse_body(int64_t raw_bytes_ptr) {
    char *buf = (char *)(uintptr_t)raw_bytes_ptr;
    int32_t len = *(int32_t *)buf;
    const char *data = buf + 4;

    // Find "\r\n\r\n".
    for (int i = 0; i + 3 < len; i++) {
        if (data[i] == '\r' && data[i+1] == '\n' &&
            data[i+2] == '\r' && data[i+3] == '\n') {
            int body_start = i + 4;
            return make_lumen_string(data + body_start, len - body_start);
        }
    }
    // No body found - return empty string.
    return make_lumen_string("", 0);
}

// Format an HTTP/1.1 response: "HTTP/1.1 {status} OK\r\nContent-Length: {len}\r\n\r\n{body}"
// body_str_ptr points to [len:i32 | data...] (Lumen string layout).
// Returns an rc_alloc'd bytes pointer.
int64_t lumen_http_format_response(int32_t status, int64_t body_str_ptr) {
    char *body_buf = (char *)(uintptr_t)body_str_ptr;
    int32_t body_len = *(int32_t *)body_buf;
    const char *body_data = body_buf + 4;

    // Convert status and body_len to strings.
    char status_str[16];
    int status_str_len = snprintf(status_str, sizeof(status_str), "%d", status);
    char len_str[16];
    int len_str_len = snprintf(len_str, sizeof(len_str), "%d", body_len);

    // Build: "HTTP/1.1 " + status + " OK\r\nContent-Length: " + len + "\r\n\r\n" + body
    const char *p1 = "HTTP/1.1 ";           int p1_len = 9;
    const char *p2 = " OK\r\nContent-Length: "; int p2_len = 20;
    const char *p3 = "\r\n\r\n";            int p3_len = 4;

    int total = p1_len + status_str_len + p2_len + len_str_len + p3_len + body_len;
    char *payload = alloc_bytes(total);
    char *out = payload + 4; // skip past len field (already set by alloc_bytes)

    memcpy(out, p1, p1_len);                out += p1_len;
    memcpy(out, status_str, status_str_len); out += status_str_len;
    memcpy(out, p2, p2_len);                out += p2_len;
    memcpy(out, len_str, len_str_len);       out += len_str_len;
    memcpy(out, p3, p3_len);                out += p3_len;
    if (body_len > 0) memcpy(out, body_data, body_len);

    return (int64_t)(uintptr_t)payload;
}

// --- Dynamic list (List<T>) --------------------------------------------------
// Layout: [rc:i32 | magic:i32 | len:i32 | cap:i32 | elem_size:i32 | pad:i32 | data...]
// The pointer Lumen sees points past the 8-byte rc header to [len | cap | elem_size | pad | data].
// Elements are stored contiguously starting at offset 16 (past the 4 metadata i32s).

typedef struct {
    int32_t len;
    int32_t cap;
    int32_t elem_size;
    int32_t _pad;
    // data follows inline
} ListHeader;

#define LIST_DATA(hdr) ((char *)(hdr) + sizeof(ListHeader))

int64_t lumen_list_new(int32_t elem_size) {
    int32_t initial_cap = 8;
    int32_t total = sizeof(ListHeader) + elem_size * initial_cap;
    ListHeader *hdr = (ListHeader *)current_allocator->alloc(
        current_allocator->state, (int64_t)total);
    hdr->len = 0;
    hdr->cap = initial_cap;
    hdr->elem_size = elem_size;
    hdr->_pad = 0;
    return (int64_t)(uintptr_t)hdr;
}

int32_t lumen_list_len(int64_t list_ptr) {
    ListHeader *hdr = (ListHeader *)(uintptr_t)list_ptr;
    return hdr->len;
}

// Grow the list if needed, returning the (possibly new) pointer.
static int64_t list_ensure_cap(int64_t list_ptr) {
    ListHeader *hdr = (ListHeader *)(uintptr_t)list_ptr;
    if (hdr->len < hdr->cap) return list_ptr;
    // Guard against int32 overflow on capacity doubling.
    if (hdr->cap > INT32_MAX / 2) {
        fprintf(stderr, "FATAL: list capacity overflow (cap=%d)\n", hdr->cap);
        abort();
    }
    int32_t old_cap = hdr->cap;
    int32_t new_cap = old_cap * 2;
    int32_t old_total = (int32_t)sizeof(ListHeader) + hdr->elem_size * old_cap;
    int32_t new_total = (int32_t)sizeof(ListHeader) + hdr->elem_size * new_cap;
    hdr = (ListHeader *)lumen_realloc_raw(hdr, old_total, new_total);
    hdr->cap = new_cap;
    return (int64_t)(uintptr_t)hdr;
}

// Push an i64-sized element (covers i32, i64, f64, pointers).
int64_t lumen_list_push(int64_t list_ptr, int64_t value) {
    list_ptr = list_ensure_cap(list_ptr);
    ListHeader *hdr = (ListHeader *)(uintptr_t)list_ptr;
    char *slot = LIST_DATA(hdr) + (int64_t)hdr->len * hdr->elem_size;
    if (hdr->elem_size == 4) {
        *(int32_t *)slot = (int32_t)value;
    } else {
        *(int64_t *)slot = value;
    }
    hdr->len++;
    return list_ptr;  // may have moved due to realloc
}

// Get element at index. Returns i64 (caller truncates).
int64_t lumen_list_get(int64_t list_ptr, int32_t index) {
    ListHeader *hdr = (ListHeader *)(uintptr_t)list_ptr;
    if (index < 0 || index >= hdr->len) return 0;
    char *slot = LIST_DATA(hdr) + (int64_t)index * hdr->elem_size;
    if (hdr->elem_size == 4) {
        return (int64_t)*(int32_t *)slot;
    } else {
        return *(int64_t *)slot;
    }
}

// Set element at index. Returns the list pointer (unchanged).
int64_t lumen_list_set(int64_t list_ptr, int32_t index, int64_t value) {
    ListHeader *hdr = (ListHeader *)(uintptr_t)list_ptr;
    if (index < 0 || index >= hdr->len) return list_ptr;
    char *slot = LIST_DATA(hdr) + (int64_t)index * hdr->elem_size;
    if (hdr->elem_size == 4) {
        *(int32_t *)slot = (int32_t)value;
    } else {
        *(int64_t *)slot = value;
    }
    return list_ptr;
}

// Remove element at index, shifting remaining elements left.
int64_t lumen_list_remove(int64_t list_ptr, int32_t index) {
    ListHeader *hdr = (ListHeader *)(uintptr_t)list_ptr;
    if (index < 0 || index >= hdr->len) return list_ptr;
    char *data = LIST_DATA(hdr);
    int32_t es = hdr->elem_size;
    memmove(data + (int64_t)index * es,
            data + (int64_t)(index + 1) * es,
            (int64_t)(hdr->len - index - 1) * es);
    hdr->len--;
    return list_ptr;
}

// --- Green thread runtime (ucontext + epoll) --------------------------------

#define GT_MAX 4096
#define GT_STACK_SIZE (64 * 1024)

typedef enum { GT_FREE, GT_RUNNABLE, GT_BLOCKED, GT_DONE } GTState;

typedef struct {
    ucontext_t ctx;
    char *stack;
    GTState state;
    int wait_fd;
} GreenThread;

// Per-thread green-thread pool. Each OS thread carries its own
// cooperative scheduler, so I/O blocking in a green thread on worker
// A doesn't stall actors on worker B. `__thread` is straightforward
// here — the gt_* statics are self-contained per-worker state.
static __thread GreenThread gt_pool[GT_MAX];
static __thread int gt_count = 0;
static __thread int gt_current = -1;
static __thread ucontext_t gt_sched_ctx;
static __thread int gt_event_fd = -1;

static void gt_init(void) {
    if (gt_event_fd < 0) {
#ifdef __linux__
        gt_event_fd = epoll_create1(0);
#elif defined(__APPLE__)
        gt_event_fd = kqueue();
#endif
        for (int i = 0; i < GT_MAX; i++) gt_pool[i].state = GT_FREE;
    }
}

typedef struct { void (*fn)(int); int arg; } GTEntry;

static void gt_entry(unsigned int lo, unsigned int hi) {
    GTEntry *e = (GTEntry *)(uintptr_t)((uint64_t)lo | ((uint64_t)hi << 32));
    e->fn(e->arg);
    free(e);
    gt_pool[gt_current].state = GT_DONE;
    swapcontext(&gt_pool[gt_current].ctx, &gt_sched_ctx);
}

// Returns 0 on success, -1 if no free thread slot.
static int gt_spawn(void (*fn)(int), int arg) {
    gt_init();
    int id = -1;
    for (int i = 0; i < GT_MAX; i++) {
        if (gt_pool[i].state == GT_FREE || gt_pool[i].state == GT_DONE) {
            id = i; break;
        }
    }
    if (id < 0) return -1;

    if (gt_pool[id].stack) free(gt_pool[id].stack);
    gt_pool[id].stack = malloc(GT_STACK_SIZE);
    gt_pool[id].state = GT_RUNNABLE;
    gt_pool[id].wait_fd = -1;

    GTEntry *entry = malloc(sizeof(GTEntry));
    entry->fn = fn;
    entry->arg = arg;

    getcontext(&gt_pool[id].ctx);
    gt_pool[id].ctx.uc_stack.ss_sp = gt_pool[id].stack;
    gt_pool[id].ctx.uc_stack.ss_size = GT_STACK_SIZE;
    gt_pool[id].ctx.uc_link = NULL;
    uintptr_t ptr = (uintptr_t)entry;
    makecontext(&gt_pool[id].ctx, (void (*)())gt_entry,
                2, (unsigned int)(ptr & 0xFFFFFFFF), (unsigned int)(ptr >> 32));
    if (id >= gt_count) gt_count = id + 1;
    return 0;
}

// Portable event flag constants for callsites below.
#ifdef __linux__
#define GT_EV_READ  EPOLLIN
#define GT_EV_WRITE EPOLLOUT
#else
#define GT_EV_READ  0x001
#define GT_EV_WRITE 0x004
#endif

// Block current green thread until fd is ready.
static void gt_wait_fd(int fd, int events) {
    if (gt_current < 0) return;
#ifdef __linux__
    struct epoll_event ev = { .events = events | EPOLLONESHOT,
                              .data.fd = gt_current };
    epoll_ctl(gt_event_fd, EPOLL_CTL_ADD, fd, &ev);
#elif defined(__APPLE__)
    int16_t filter = (events & GT_EV_WRITE) ? EVFILT_WRITE : EVFILT_READ;
    struct kevent kev;
    EV_SET(&kev, fd, filter, EV_ADD | EV_ONESHOT, 0, 0, (void *)(uintptr_t)gt_current);
    kevent(gt_event_fd, &kev, 1, NULL, 0, NULL);
#endif
    gt_pool[gt_current].state = GT_BLOCKED;
    gt_pool[gt_current].wait_fd = fd;
    swapcontext(&gt_pool[gt_current].ctx, &gt_sched_ctx);
#ifdef __linux__
    epoll_ctl(gt_event_fd, EPOLL_CTL_DEL, fd, NULL);
#endif
}

static void gt_schedule(void) {
    while (1) {
        int ran = 0;
        for (int i = 0; i < gt_count; i++) {
            if (gt_pool[i].state == GT_RUNNABLE) {
                gt_current = i;
                swapcontext(&gt_sched_ctx, &gt_pool[i].ctx);
                gt_current = -1;
                ran = 1;
                break;
            }
        }
        if (ran) continue;

        int alive = 0;
        for (int i = 0; i < gt_count; i++)
            if (gt_pool[i].state == GT_BLOCKED) alive++;
        if (alive == 0) break;

#ifdef __linux__
        struct epoll_event events[64];
        int n = epoll_wait(gt_event_fd, events, 64, 100);
        for (int i = 0; i < n; i++) {
            int tid = events[i].data.fd;
            if (tid >= 0 && tid < gt_count && gt_pool[tid].state == GT_BLOCKED)
                gt_pool[tid].state = GT_RUNNABLE;
        }
#elif defined(__APPLE__)
        struct kevent events[64];
        struct timespec timeout = { .tv_sec = 0, .tv_nsec = 100000000 };
        int n = kevent(gt_event_fd, NULL, 0, events, 64, &timeout);
        for (int i = 0; i < n; i++) {
            int tid = (int)(uintptr_t)events[i].udata;
            if (tid >= 0 && tid < gt_count && gt_pool[tid].state == GT_BLOCKED)
                gt_pool[tid].state = GT_RUNNABLE;
        }
#endif
    }
}

// --- net.serve: green-thread-per-connection HTTP server ----------------------

static void (*gt_handler)(int);

static void gt_connection_handler(int client_fd) {
    gt_handler(client_fd);
    close(client_fd);
}

static void gt_accept_loop(int server_fd) {
    fcntl(server_fd, F_SETFL, fcntl(server_fd, F_GETFL) | O_NONBLOCK);
    while (1) {
        int client = accept(server_fd, NULL, NULL);
        if (client < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                gt_wait_fd(server_fd, GT_EV_READ);
                continue;
            }
            break;
        }
        fcntl(client, F_SETFL, fcntl(client, F_GETFL) | O_NONBLOCK);
        if (gt_spawn(gt_connection_handler, client) < 0) {
            fprintf(stderr, "warning: all green thread slots full, dropping connection\n");
            close(client);
        }
    }
}

void lumen_net_serve(int port, void (*handler)(int)) {
    gt_init();
    gt_handler = handler;
    int server_fd = lumen_tcp_listen(port);
    if (server_fd < 0) {
        fprintf(stderr, "net.serve: failed to listen on port %d\n", port);
        return;
    }
    fprintf(stderr, "Listening on :%d (green threads)\n", port);
    gt_spawn(gt_accept_loop, server_fd);
    gt_schedule();
}

// Green-thread-aware read.
int64_t lumen_gt_read(int fd, int max) {
    char *buf = alloc_bytes(max);
    if (!buf) return 0;
    while (1) {
        ssize_t n = read(fd, buf + 4, max);
        if (n >= 0) {
            *(int32_t *)buf = (int32_t)n;
            return (int64_t)(uintptr_t)buf;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            gt_wait_fd(fd, GT_EV_READ);
            continue;
        }
        *(int32_t *)buf = 0;
        return (int64_t)(uintptr_t)buf;
    }
}

// Green-thread-aware write.
int32_t lumen_gt_write(int fd, int64_t bytes_ptr) {
    char *buf = (char *)(uintptr_t)bytes_ptr;
    int32_t len = *(int32_t *)buf;
    int32_t written = 0;
    while (written < len) {
        ssize_t n = write(fd, buf + 4 + written, len - written);
        if (n >= 0) {
            written += n;
        } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
            gt_wait_fd(fd, GT_EV_WRITE);
        } else {
            break;
        }
    }
    return (int32_t)written;
}

// --- Math helpers --------------------------------------------------------

double lumen_sqrt(double x) { return sqrt(x); }
double lumen_abs(double x) { return fabs(x); }
double lumen_cos(double x) { return cos(x); }
double lumen_sin(double x) { return sin(x); }
double lumen_clamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}
double lumen_rand_f64(void) { return (double)rand() / (double)RAND_MAX; }

// --- String helpers (println, itoa, concat) ------------------------------

void lumen_println(int64_t str_ptr) {
    if (str_ptr == 0) { write(1, "\n", 1); return; }
    char *buf = (char *)(uintptr_t)str_ptr;
    int32_t len = *(int32_t *)buf;
    write(1, buf + 4, len);
    write(1, "\n", 1);
}

// Convert i32 to Lumen string [len:i32 | digits].
int64_t lumen_itoa(int32_t n) {
    char tmp[16];
    int neg = 0;
    int32_t v = n;
    if (v < 0) { neg = 1; v = -v; }
    int pos = 0;
    if (v == 0) { tmp[pos++] = '0'; }
    else { while (v > 0) { tmp[pos++] = '0' + (v % 10); v /= 10; } }
    if (neg) tmp[pos++] = '-';
    // Reverse into a Lumen string.
    int32_t len = pos;
    char *payload = (char *)current_allocator->alloc(
        current_allocator->state, (int64_t)(4 + len));
    *(int32_t *)payload = len;
    for (int i = 0; i < len; i++) {
        payload[4 + i] = tmp[len - 1 - i];
    }
    return (int64_t)(uintptr_t)payload;
}

// Concatenate two Lumen strings → new Lumen string.
int64_t lumen_concat(int64_t a_ptr, int64_t b_ptr) {
    char *a = (char *)(uintptr_t)a_ptr;
    char *b = (char *)(uintptr_t)b_ptr;
    int32_t a_len = a ? *(int32_t *)a : 0;
    int32_t b_len = b ? *(int32_t *)b : 0;
    int32_t total = a_len + b_len;
    char *payload = (char *)current_allocator->alloc(
        current_allocator->state, (int64_t)(4 + total));
    *(int32_t *)payload = total;
    if (a_len > 0) memcpy(payload + 4, a + 4, a_len);
    if (b_len > 0) memcpy(payload + 4 + a_len, b + 4, b_len);
    return (int64_t)(uintptr_t)payload;
}

// --- debug.print primitives (output to stderr) ---------------------------

// --- Crash handler for --debug mode --------------------------------------

// The frame chain is a global linked list. We declare it as extern here
// since it's defined in the Cranelift-generated object.
extern char *lumen_frame_chain;

// Print a backtrace via libc's backtrace() + backtrace_symbols_fd().
// Walks the DWARF .eh_frame we emit by default, so function names
// from .symtab show up without requiring --debug. Addresses can be
// fed through `addr2line -e <binary>` to get file:line locations
// from the .debug_line table.
#include <execinfo.h>
static void lumen_print_backtrace(void) {
    void *buf[64];
    int n = backtrace(buf, 64);
    if (n > 0) {
        backtrace_symbols_fd(buf, n, 2);  // stderr
    }
}

static void lumen_crash_handler(int sig) {
    const char *name = sig == SIGSEGV ? "SIGSEGV" : sig == SIGABRT ? "SIGABRT" : "signal";
    fprintf(stderr, "\n--- CRASH: %s ---\nStack trace:\n", name);
    lumen_print_backtrace();
    // If --debug mode is on, also dump the push/pop message stack
    // for human-readable context (annotated with file:col).
    void lumen_debug_print_stack(void);
    lumen_debug_print_stack();
    fprintf(stderr, "---\n");
    _exit(128 + sig);
}

// Install the default crash handler at program start — runs via the
// ELF .init_array before main, so any SIGSEGV/SIGABRT in user code
// gets an eh_frame-backed backtrace even without --debug.
__attribute__((constructor))
static void lumen_install_default_crash_handler(void) {
    signal(SIGSEGV, lumen_crash_handler);
    signal(SIGABRT, lumen_crash_handler);
}

// --debug mode keeps lumen_debug_init as a no-op shim for backwards
// compat — the default installer above already wired things up.
void lumen_debug_init(void) {
    // No-op: the constructor above handles this.
}

// Debug frame stack: fixed-size array of message pointers.
// Much cheaper than linked-list allocation — just a pointer bump.
#define DEBUG_STACK_MAX 4096
// Per-thread debug frame stack. Each worker's scope-push/pop for
// --debug mode is isolated — a worker crashing prints only its own
// frames, not the aggregate across threads.
static __thread int64_t debug_stack[DEBUG_STACK_MAX];
static __thread int32_t debug_stack_top = 0;

void lumen_debug_push(int64_t msg_ptr) {
    if (debug_stack_top < DEBUG_STACK_MAX) {
        debug_stack[debug_stack_top++] = msg_ptr;
    }
}

void lumen_debug_pop(void) {
    if (debug_stack_top > 0) debug_stack_top--;
}

// Print the debug stack (called by crash handler instead of frame_chain).
void lumen_debug_print_stack(void) {
    for (int i = debug_stack_top - 1; i >= 0; i--) {
        char *msg = (char *)(uintptr_t)debug_stack[i];
        if (msg) {
            int32_t len = *(int32_t *)msg;
            fwrite(msg + 4, 1, len, stderr);
            fputc('\n', stderr);
        }
    }
}

// --- debug.print primitives (output to stderr) ---------------------------

// String content comparison: returns 1 if equal, 0 if not.
int32_t lumen_string_eq(int64_t a_ptr, int64_t b_ptr) {
    if (a_ptr == b_ptr) return 1;  // same pointer → equal
    if (a_ptr == 0 || b_ptr == 0) return a_ptr == b_ptr;
    char *a = (char *)(uintptr_t)a_ptr;
    char *b = (char *)(uintptr_t)b_ptr;
    int32_t a_len = *(int32_t *)a;
    int32_t b_len = *(int32_t *)b;
    if (a_len != b_len) return 0;
    return memcmp(a + 4, b + 4, a_len) == 0 ? 1 : 0;
}

void lumen_debug_i32(int32_t v) { fprintf(stderr, "%d", v); }
void lumen_debug_i64(int64_t v) { fprintf(stderr, "%lld", (long long)v); }
void lumen_debug_f64(double v) { fprintf(stderr, "%g", v); }
void lumen_debug_bool(int32_t v) { fprintf(stderr, v ? "true" : "false"); }
void lumen_debug_str(int64_t ptr) {
    if (ptr == 0) { fprintf(stderr, "\"\""); return; }
    char *buf = (char *)(uintptr_t)ptr;
    int32_t len = *(int32_t *)buf;
    fprintf(stderr, "\"");
    fwrite(buf + 4, 1, len, stderr);
    fprintf(stderr, "\"");
}
void lumen_debug_raw(const char *s, int32_t len) { fwrite(s, 1, len, stderr); }
void lumen_debug_newline(void) { fprintf(stderr, "\n"); }

// --- io.println formatting primitives (output to stdout) -----------------
// Mirror the debug.* primitives but write to stdout and DO NOT wrap strings
// in quotes (println should print the string content directly).
void lumen_io_i32(int32_t v) { fprintf(stdout, "%d", v); }
void lumen_io_i64(int64_t v) { fprintf(stdout, "%lld", (long long)v); }
void lumen_io_f64(double v) { fprintf(stdout, "%g", v); }
void lumen_io_bool(int32_t v) { fputs(v ? "true" : "false", stdout); }
void lumen_io_str(int64_t ptr) {
    if (ptr == 0) return;
    char *buf = (char *)(uintptr_t)ptr;
    int32_t len = *(int32_t *)buf;
    fwrite(buf + 4, 1, len, stdout);
}
void lumen_io_raw(const char *s, int32_t len) { fwrite(s, 1, len, stdout); }
// Flush in newline so io.println output stays in order with lumen_println
// (which writes via raw write(1, ...) and bypasses stdio buffering).
// Without this, when stdout is a pipe (e.g. piped into a test harness),
// fprintf-buffered io_* output and write-direct lumen_println output
// interleave out of order.
void lumen_io_newline(void) { fputc('\n', stdout); fflush(stdout); }

// --- String-buffer helpers (string interpolation) ------------------------
// A growable byte buffer used to assemble interpolated strings. After all
// pieces are appended, lumen_strbuf_finish converts it into a Lumen string
// ([rc:i32 | magic:i32 | len:i32 | bytes]) and frees the buffer.

typedef struct {
    char *data;
    int32_t len;
    int32_t cap;
} LumenStrBuf;

void *lumen_strbuf_new(void) {
    LumenStrBuf *b = (LumenStrBuf *)malloc(sizeof(LumenStrBuf));
    b->cap = 64;
    b->len = 0;
    b->data = (char *)malloc(b->cap);
    return b;
}

static void lumen_strbuf_grow(LumenStrBuf *b, int32_t need) {
    if (b->len + need <= b->cap) return;
    while (b->cap < b->len + need) b->cap *= 2;
    b->data = (char *)realloc(b->data, b->cap);
}

void lumen_strbuf_raw(void *buf, const char *s, int32_t len) {
    LumenStrBuf *b = (LumenStrBuf *)buf;
    lumen_strbuf_grow(b, len);
    memcpy(b->data + b->len, s, len);
    b->len += len;
}

void lumen_strbuf_str(void *buf, int64_t str_ptr) {
    if (str_ptr == 0) return;
    char *s = (char *)(uintptr_t)str_ptr;
    int32_t len = *(int32_t *)s;
    lumen_strbuf_raw(buf, s + 4, len);
}

void lumen_strbuf_i32(void *buf, int32_t v) {
    char tmp[16];
    int n = snprintf(tmp, sizeof(tmp), "%d", v);
    lumen_strbuf_raw(buf, tmp, n);
}

void lumen_strbuf_i64(void *buf, int64_t v) {
    char tmp[24];
    int n = snprintf(tmp, sizeof(tmp), "%lld", (long long)v);
    lumen_strbuf_raw(buf, tmp, n);
}

void lumen_strbuf_f64(void *buf, double v) {
    char tmp[32];
    int n = snprintf(tmp, sizeof(tmp), "%g", v);
    lumen_strbuf_raw(buf, tmp, n);
}

void lumen_strbuf_bool(void *buf, int32_t v) {
    if (v) lumen_strbuf_raw(buf, "true", 4);
    else lumen_strbuf_raw(buf, "false", 5);
}

int64_t lumen_strbuf_finish(void *buf) {
    LumenStrBuf *b = (LumenStrBuf *)buf;
    int32_t total = b->len;
    char *payload = (char *)current_allocator->alloc(
        current_allocator->state, (int64_t)(4 + total));
    *(int32_t *)payload = total;
    if (total > 0) memcpy(payload + 4, b->data, total);
    free(b->data);
    free(b);
    return (int64_t)(uintptr_t)payload;
}

// --- assert builtin ------------------------------------------------------
// cond: 0 = fail, nonzero = pass.
// msg_ptr: optional Lumen string (0 if absent).
// file_ptr: Lumen string of the source path (interned at codegen time).
// line, col: 1-based source location.
// debug_mode: 1 if compiled with --debug (also dumps the frame stack).
void lumen_assert(int32_t cond, int64_t msg_ptr, int64_t file_ptr,
                  int32_t line, int32_t col, int32_t debug_mode) {
    if (cond) return;
    fprintf(stderr, "assertion failed at ");
    if (file_ptr != 0) {
        char *file = (char *)(uintptr_t)file_ptr;
        int32_t file_len = *(int32_t *)file;
        fwrite(file + 4, 1, file_len, stderr);
    } else {
        fprintf(stderr, "<unknown>");
    }
    fprintf(stderr, ":%d:%d", line, col);
    if (msg_ptr != 0) {
        char *msg = (char *)(uintptr_t)msg_ptr;
        int32_t msg_len = *(int32_t *)msg;
        fprintf(stderr, ": ");
        fwrite(msg + 4, 1, msg_len, stderr);
    }
    fprintf(stderr, "\n");
    // Always emit a backtrace on assert — libc's backtrace() uses
    // the .eh_frame we emit by default, giving function names +
    // addresses even without --debug. --debug adds the annotated
    // push/pop message stack on top.
    fprintf(stderr, "Stack trace:\n");
    lumen_print_backtrace();
    if (debug_mode) {
        lumen_debug_print_stack();
    }
    abort();
}

// --- File I/O ------------------------------------------------------------
// All errors are reported as POSIX errno values; 0 means success.
//
// lumen_fs_read(path) returns the file contents as an rc_alloc'd Lumen
// string (rc=1) on success, or 0 on failure. After every call, the most
// recent errno is available via lumen_fs_errno() — the std/fs Lumen
// wrapper checks this immediately and turns the pair into a Result.
//
// Single-OS-thread assumption: Lumen runs on cooperative green threads
// today, so a static is safe. Switch to thread_local if/when that changes.

// Per-thread errno sidechannel: each worker that reads/writes files
// observes only its own errors. Without __thread, one actor's fs
// failure would corrupt another's lumen_fs_errno() result.
static __thread int32_t lumen_last_fs_errno = 0;

int32_t lumen_fs_errno(void) { return lumen_last_fs_errno; }

// Copy a Lumen string into a NUL-terminated C path buffer. Returns 0 on
// success, sets errno and returns -1 if the path is missing or too long.
static int lumen_fs_path_to_cstr(int64_t path_ptr, char *out, size_t out_size) {
    if (path_ptr == 0) { errno = EINVAL; return -1; }
    char *src = (char *)(uintptr_t)path_ptr;
    int32_t len = *(int32_t *)src;
    if (len < 0 || (size_t)len + 1 > out_size) { errno = ENAMETOOLONG; return -1; }
    memcpy(out, src + 4, (size_t)len);
    out[len] = 0;
    return 0;
}

int64_t lumen_fs_read(int64_t path_ptr) {
    char path[4096];
    if (lumen_fs_path_to_cstr(path_ptr, path, sizeof(path)) != 0) {
        lumen_last_fs_errno = errno;
        return 0;
    }
    FILE *f = fopen(path, "rb");
    if (!f) { lumen_last_fs_errno = errno; return 0; }
    if (fseek(f, 0, SEEK_END) != 0) { lumen_last_fs_errno = errno; fclose(f); return 0; }
    long size = ftell(f);
    if (size < 0) { lumen_last_fs_errno = errno; fclose(f); return 0; }
    rewind(f);

    char *payload = (char *)current_allocator->alloc(
        current_allocator->state, (int64_t)(4 + (size_t)size));
    *(int32_t *)payload = (int32_t)size;
    if (size > 0) {
        size_t got = fread(payload + 4, 1, (size_t)size, f);
        if (got != (size_t)size) {
            int e = ferror(f) ? errno : EIO;
            // Leak the partial allocation — on malloc-backed it's a
            // real leak, on arena it's reclaimed at arena close.
            fclose(f);
            lumen_last_fs_errno = e ? e : EIO;
            return 0;
        }
    }
    fclose(f);
    lumen_last_fs_errno = 0;
    return (int64_t)(uintptr_t)payload;
}

// --- HashMap ------------------------------------------------------------
// Insertion-ordered hash map (Python compact-dict design).
//
// One inline rc-allocated block:
//   [rc:i32 | magic:i32]
//   [MapHeader (32 bytes)]
//   [entries[entry_cap]]   — dense, insertion-ordered; tombstoned entries
//                            keep their slot (key_ptr=0) until the next resize
//   [index[index_cap]]      — sparse table; each slot holds an i32 entry
//                            index, or -1 (empty) / -2 (tombstone)
//
// Lookups: hash → walk index[] with linear probing, comparing entry keys.
// Iteration: walk entries[] in order, skipping tombstones (caller-side).
// Resize: realloc the whole block, rebuild index from entries (also
// compacts entries by dropping tombstones).
//
// Keys are Lumen strings (MVP). Values are i64-shaped (caller widens
// i32→i64 on the way in; codegen ireduces on the way out, like List).
//
// On overwrite (map_set with an existing key) and on map_remove the
// displaced/removed key is rc_decr'd, and the displaced/removed value is
// rc_decr'd when the caller passes value_is_ptr=1. rc_decr no-ops on
// non-rc-allocated pointers (string literals, etc.) via its magic check,
// so this is safe regardless of where the key came from.

// Declared (and exported) by the Cranelift-generated module.
extern void lumen_rc_decr(int64_t ptr);

#define LUMEN_MAP_INDEX_EMPTY (-1)
#define LUMEN_MAP_INDEX_TOMB  (-2)

typedef struct {
    // For pointer keys: Lumen string/bytes ptr. For scalar keys: the
    // key's 8-byte representation (i32 sign-extended to i64).
    // Tombstone state is tracked by `alive` — key == 0 is a valid
    // scalar key.
    int64_t key;
    int64_t value;     // i64-shaped value
    uint32_t hash;
    uint32_t alive;    // 0 = empty/tombstone, 1 = live
} LumenMapEntry;

typedef struct {
    int32_t live_count;     // valid entries (excludes tombstones)
    int32_t entry_count;    // total used in entries[] (incl tombstones)
    int32_t entry_cap;      // capacity of entries[]
    int32_t index_cap;      // capacity of index[] (power of 2)
    // 1 when keys are pointer-shaped (string/bytes); 0 for scalar
    // keys (i32/i64/char/bool). Set at map.new time and treated as
    // immutable — mixing key kinds in one map is undefined.
    int32_t key_is_ptr;
    int32_t _pad[3];        // pad to 32 bytes for alignment
} LumenMapHeader;

#define LUMEN_MAP_ENTRIES(hdr) \
    ((LumenMapEntry *)((char *)(hdr) + sizeof(LumenMapHeader)))
#define LUMEN_MAP_INDEX(hdr) \
    ((int32_t *)((char *)LUMEN_MAP_ENTRIES(hdr) + (size_t)(hdr)->entry_cap * sizeof(LumenMapEntry)))

// Per-thread sidechannel for map.get's "did we find it?" result.
// The companion signature (see lumen_map_get_found / lumen_map_get
// in std/map.lm) reads this immediately after the call; __thread
// keeps concurrent map.get calls from different workers isolated.
static __thread int32_t lumen_map_last_get_found = 0;

// FNV-1a 32-bit. Cheap, decent distribution for short string keys and
// for the 8 bytes of a scalar key.
static uint32_t lumen_map_hash_bytes(const uint8_t *data, int32_t len) {
    uint32_t h = 2166136261u;
    for (int32_t i = 0; i < len; i++) {
        h ^= data[i];
        h *= 16777619u;
    }
    return h;
}

static uint32_t lumen_map_hash_key(int64_t key, int32_t key_is_ptr) {
    if (key_is_ptr) {
        if (key == 0) return 0;
        char *kbuf = (char *)(uintptr_t)key;
        int32_t klen = *(int32_t *)kbuf;
        return lumen_map_hash_bytes((const uint8_t *)(kbuf + 4), klen);
    }
    return lumen_map_hash_bytes((const uint8_t *)&key, 8);
}

static int lumen_map_keys_equal(int64_t a, int64_t b, int32_t key_is_ptr) {
    if (!key_is_ptr) return a == b;
    if (a == 0 || b == 0) return 0;
    char *abuf = (char *)(uintptr_t)a;
    char *bbuf = (char *)(uintptr_t)b;
    int32_t alen = *(int32_t *)abuf;
    int32_t blen = *(int32_t *)bbuf;
    return alen == blen && memcmp(abuf + 4, bbuf + 4, (size_t)alen) == 0;
}

// Lookup: returns the index[] position for the key (existing or insertion
// point). *out_entry is the entry index if found, -1 if not. Bounded to
// index_cap probes so a fully tombstoned table can't loop forever (the
// resize logic in lumen_map_set normally prevents this, but a sequence of
// removes without an intervening set could still saturate the index).
static int32_t lumen_map_find_slot(LumenMapHeader *hdr, int64_t key,
                                   uint32_t hash, int32_t *out_entry) {
    int32_t mask = hdr->index_cap - 1;
    int32_t pos = (int32_t)(hash & (uint32_t)mask);
    int32_t first_tomb = -1;
    int32_t *index = LUMEN_MAP_INDEX(hdr);
    LumenMapEntry *entries = LUMEN_MAP_ENTRIES(hdr);
    for (int32_t probe = 0; probe < hdr->index_cap; probe++) {
        int32_t slot = index[pos];
        if (slot == LUMEN_MAP_INDEX_EMPTY) {
            *out_entry = -1;
            return first_tomb >= 0 ? first_tomb : pos;
        }
        if (slot == LUMEN_MAP_INDEX_TOMB) {
            if (first_tomb < 0) first_tomb = pos;
        } else {
            LumenMapEntry *e = &entries[slot];
            if (e->alive && e->hash == hash
                && lumen_map_keys_equal(e->key, key, hdr->key_is_ptr)) {
                *out_entry = slot;
                return pos;
            }
        }
        pos = (pos + 1) & mask;
    }
    // Probed the entire index without finding the key or an empty slot.
    *out_entry = -1;
    return first_tomb >= 0 ? first_tomb : 0;
}

int64_t lumen_map_new(int32_t key_is_ptr) {
    int32_t entry_cap = 8;
    int32_t index_cap = 16;
    size_t total = sizeof(LumenMapHeader)
                 + (size_t)entry_cap * sizeof(LumenMapEntry)
                 + (size_t)index_cap * sizeof(int32_t);
    LumenMapHeader *hdr = (LumenMapHeader *)current_allocator->alloc(
        current_allocator->state, (int64_t)total);
    hdr->live_count = 0;
    hdr->entry_count = 0;
    hdr->entry_cap = entry_cap;
    hdr->index_cap = index_cap;
    hdr->key_is_ptr = key_is_ptr;
    hdr->_pad[0] = 0;
    hdr->_pad[1] = 0;
    hdr->_pad[2] = 0;
    int32_t *index = LUMEN_MAP_INDEX(hdr);
    for (int32_t i = 0; i < index_cap; i++) index[i] = LUMEN_MAP_INDEX_EMPTY;
    return (int64_t)(uintptr_t)hdr;
}

// realloc the block to new sizes, then compact entries (drop tombstones)
// and rebuild the index from scratch.
static int64_t lumen_map_grow(int64_t map_ptr, int32_t new_entry_cap, int32_t new_index_cap) {
    LumenMapHeader *hdr = (LumenMapHeader *)(uintptr_t)map_ptr;
    size_t old_total = sizeof(LumenMapHeader)
                     + (size_t)hdr->entry_cap * sizeof(LumenMapEntry)
                     + (size_t)hdr->index_cap * sizeof(int32_t);
    size_t new_total = sizeof(LumenMapHeader)
                     + (size_t)new_entry_cap * sizeof(LumenMapEntry)
                     + (size_t)new_index_cap * sizeof(int32_t);
    hdr = (LumenMapHeader *)lumen_realloc_raw(hdr, (int64_t)old_total, (int64_t)new_total);
    int32_t old_entry_count = hdr->entry_count;
    hdr->entry_cap = new_entry_cap;
    hdr->index_cap = new_index_cap;

    LumenMapEntry *entries = LUMEN_MAP_ENTRIES(hdr);
    int32_t *index = LUMEN_MAP_INDEX(hdr);
    for (int32_t i = 0; i < new_index_cap; i++) index[i] = LUMEN_MAP_INDEX_EMPTY;

    int32_t mask = new_index_cap - 1;
    int32_t write_idx = 0;
    int32_t live = 0;
    for (int32_t i = 0; i < old_entry_count; i++) {
        if (!entries[i].alive) continue;  // tombstone
        if (write_idx != i) entries[write_idx] = entries[i];
        uint32_t h = entries[write_idx].hash;
        int32_t pos = (int32_t)(h & (uint32_t)mask);
        while (index[pos] != LUMEN_MAP_INDEX_EMPTY) pos = (pos + 1) & mask;
        index[pos] = write_idx;
        write_idx++;
        live++;
    }
    hdr->entry_count = write_idx;
    hdr->live_count = live;
    return (int64_t)(uintptr_t)hdr;
}

int64_t lumen_map_set(int64_t map_ptr, int64_t key, int64_t value, int32_t value_is_ptr) {
    LumenMapHeader *hdr = (LumenMapHeader *)(uintptr_t)map_ptr;
    int32_t key_is_ptr = hdr->key_is_ptr;
    // For pointer keys, 0 is the null pointer (no valid key). For
    // scalar keys, 0 is a valid key value.
    if (key_is_ptr && key == 0) return map_ptr;
    uint32_t hash = lumen_map_hash_key(key, key_is_ptr);

    int32_t need_index = ((hdr->entry_count + 1) * 4) >= (hdr->index_cap * 3);
    int32_t need_entry = hdr->entry_count >= hdr->entry_cap;
    if (need_index || need_entry) {
        int32_t new_entry_cap = need_entry ? hdr->entry_cap * 2 : hdr->entry_cap;
        int32_t new_index_cap = need_index ? hdr->index_cap * 2 : hdr->index_cap;
        map_ptr = lumen_map_grow(map_ptr, new_entry_cap, new_index_cap);
        hdr = (LumenMapHeader *)(uintptr_t)map_ptr;
    }

    int32_t found_idx;
    int32_t pos = lumen_map_find_slot(hdr, key, hash, &found_idx);
    if (found_idx >= 0) {
        // Overwrite: release the references the map held, install the new
        // ones. The caller already rc_incr'd key and value (when each is
        // pointer-typed), so the map's net refcount on each is +1.
        LumenMapEntry *e = &LUMEN_MAP_ENTRIES(hdr)[found_idx];
        if (value_is_ptr) lumen_rc_decr(e->value);
        if (key_is_ptr) lumen_rc_decr(e->key);
        e->key = key;
        e->value = value;
        return map_ptr;
    }
    int32_t new_slot = hdr->entry_count;
    LumenMapEntry *e = &LUMEN_MAP_ENTRIES(hdr)[new_slot];
    e->key = key;
    e->value = value;
    e->hash = hash;
    e->alive = 1;
    LUMEN_MAP_INDEX(hdr)[pos] = new_slot;
    hdr->entry_count++;
    hdr->live_count++;
    return map_ptr;
}

int64_t lumen_map_get(int64_t map_ptr, int64_t key) {
    LumenMapHeader *hdr = (LumenMapHeader *)(uintptr_t)map_ptr;
    int32_t key_is_ptr = hdr->key_is_ptr;
    if ((key_is_ptr && key == 0) || hdr->live_count == 0) {
        lumen_map_last_get_found = 0;
        return 0;
    }
    uint32_t hash = lumen_map_hash_key(key, key_is_ptr);
    int32_t found_idx;
    lumen_map_find_slot(hdr, key, hash, &found_idx);
    if (found_idx < 0) {
        lumen_map_last_get_found = 0;
        return 0;
    }
    lumen_map_last_get_found = 1;
    return LUMEN_MAP_ENTRIES(hdr)[found_idx].value;
}

int32_t lumen_map_get_found(void) { return lumen_map_last_get_found; }

int32_t lumen_map_contains(int64_t map_ptr, int64_t key) {
    LumenMapHeader *hdr = (LumenMapHeader *)(uintptr_t)map_ptr;
    int32_t key_is_ptr = hdr->key_is_ptr;
    if ((key_is_ptr && key == 0) || hdr->live_count == 0) return 0;
    uint32_t hash = lumen_map_hash_key(key, key_is_ptr);
    int32_t found_idx;
    lumen_map_find_slot(hdr, key, hash, &found_idx);
    return found_idx >= 0 ? 1 : 0;
}

int64_t lumen_map_remove(int64_t map_ptr, int64_t key, int32_t value_is_ptr) {
    LumenMapHeader *hdr = (LumenMapHeader *)(uintptr_t)map_ptr;
    int32_t key_is_ptr = hdr->key_is_ptr;
    if ((key_is_ptr && key == 0) || hdr->live_count == 0) return map_ptr;
    uint32_t hash = lumen_map_hash_key(key, key_is_ptr);
    int32_t found_idx;
    int32_t pos = lumen_map_find_slot(hdr, key, hash, &found_idx);
    if (found_idx < 0) return map_ptr;
    LumenMapEntry *e = &LUMEN_MAP_ENTRIES(hdr)[found_idx];
    if (value_is_ptr) lumen_rc_decr(e->value);
    if (key_is_ptr) lumen_rc_decr(e->key);
    e->alive = 0;  // tombstone
    LUMEN_MAP_INDEX(hdr)[pos] = LUMEN_MAP_INDEX_TOMB;
    hdr->live_count--;
    return map_ptr;
}

int32_t lumen_map_len(int64_t map_ptr) {
    LumenMapHeader *hdr = (LumenMapHeader *)(uintptr_t)map_ptr;
    return hdr->live_count;
}

// Walk entries[] skipping tombstones, return the key_ptr at the
// `live_i`-th live position. Returns 0 if live_i is out of range.
// Used by map.keys / map.values / map.entries to iterate insertion
// order without exposing the tombstone layout to Lumen.
int64_t lumen_map_live_key_at(int64_t map_ptr, int32_t live_i) {
    if (map_ptr == 0 || live_i < 0) return 0;
    LumenMapHeader *hdr = (LumenMapHeader *)(uintptr_t)map_ptr;
    LumenMapEntry *entries = LUMEN_MAP_ENTRIES(hdr);
    int32_t seen = 0;
    for (int32_t i = 0; i < hdr->entry_count; i++) {
        if (!entries[i].alive) continue;  // tombstone
        if (seen == live_i) return entries[i].key;
        seen++;
    }
    return 0;
}

// Value at the live_i-th live entry. Undefined if out of range; callers
// should bound `live_i` to `map_len(m)` first.
int64_t lumen_map_live_value_at(int64_t map_ptr, int32_t live_i) {
    if (map_ptr == 0 || live_i < 0) return 0;
    LumenMapHeader *hdr = (LumenMapHeader *)(uintptr_t)map_ptr;
    LumenMapEntry *entries = LUMEN_MAP_ENTRIES(hdr);
    int32_t seen = 0;
    for (int32_t i = 0; i < hdr->entry_count; i++) {
        if (!entries[i].alive) continue;
        if (seen == live_i) return entries[i].value;
        seen++;
    }
    return 0;
}

// Merge b's entries into a (last-write-wins). Returns the (possibly
// reallocated) a pointer. value_is_ptr matches lumen_map_set's
// contract: when 1, b's values are rc_incr'd on copy and any
// displaced values in a are rc_decr'd.
int64_t lumen_map_merge(int64_t a_ptr, int64_t b_ptr, int32_t value_is_ptr) {
    if (b_ptr == 0) return a_ptr;
    LumenMapHeader *b_hdr = (LumenMapHeader *)(uintptr_t)b_ptr;
    LumenMapEntry *b_entries = LUMEN_MAP_ENTRIES(b_hdr);
    int64_t out = a_ptr;
    for (int32_t i = 0; i < b_hdr->entry_count; i++) {
        if (!b_entries[i].alive) continue;
        // lumen_map_set rc_incrs the key and the value (when value_is_ptr).
        // That balances b's ref we're sharing (neither side owns
        // exclusively after the merge).
        out = lumen_map_set(out, b_entries[i].key, b_entries[i].value, value_is_ptr);
    }
    return out;
}

int32_t lumen_fs_write(int64_t path_ptr, int64_t content_ptr) {
    char path[4096];
    if (lumen_fs_path_to_cstr(path_ptr, path, sizeof(path)) != 0) {
        return errno;
    }
    FILE *f = fopen(path, "wb");
    if (!f) return errno;
    if (content_ptr != 0) {
        char *src = (char *)(uintptr_t)content_ptr;
        int32_t len = *(int32_t *)src;
        if (len > 0) {
            size_t put = fwrite(src + 4, 1, (size_t)len, f);
            if (put != (size_t)len) {
                int e = ferror(f) ? errno : EIO;
                fclose(f);
                return e ? e : EIO;
            }
        }
    }
    if (fclose(f) != 0) return errno;
    return 0;
}

// std/test counters moved into Lumen as module-level vars (lumen-1ts).
