// Lumen actor runtime: single-threaded event loop with message queue.
// Compiled to rt.o and linked with Lumen programs.

#ifdef __APPLE__
#define _XOPEN_SOURCE 600
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
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

// --- Message queue -------------------------------------------------------

typedef struct {
    void *target_cell;         // ptr to actor's mutable state cell
    void (*dispatch)(void *cell, int32_t kind, int64_t arg0, int64_t *reply);
    int32_t msg_kind;
    int64_t arg0;
    int64_t *reply_slot;       // NULL for send, &result for ask
} QueueEntry;

#define QUEUE_CAP 65536
static QueueEntry queue[QUEUE_CAP];
static int queue_head = 0;
static int queue_tail = 0;

static int queue_len(void) {
    return (queue_tail - queue_head + QUEUE_CAP) % QUEUE_CAP;
}

// Push a message onto the queue.
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

// Push a message and get a reply (blocks by draining until the reply
// is filled). For the single-threaded event loop, "blocking" means
// processing other messages until ours is handled.
int64_t lumen_rt_ask(void *cell,
                     void (*dispatch)(void *, int32_t, int64_t, int64_t *),
                     int32_t kind, int64_t arg0) {
    int64_t reply = 0;
    // Enqueue with a reply slot.
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
    // Drain until our reply is filled.
    lumen_rt_drain();
    return reply;
}

// Process one message from the queue. Returns 1 if a message was
// processed, 0 if the queue was empty.
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

// Drain the entire queue.
void lumen_rt_drain(void) {
    while (lumen_rt_step()) {}
}

// Yield point: process any pending messages. Called at function entry
// and loop headers for cooperative scheduling.
void lumen_rt_yield(void) {
    lumen_rt_step();
}

// --- TCP socket operations ------------------------------------------------

// Allocate a Lumen bytes value: [rc:i32=1 | magic:i32=0x4C554D45 | len:i32 | data...]
// Returns pointer to the payload (len field), same as lumen_rc_alloc would.
static char *alloc_bytes(int32_t len) {
    // Total: 8 (rc header) + 4 (len) + len (data)
    char *raw = (char *)malloc(8 + 4 + len);
    if (!raw) return NULL;
    *(int32_t *)(raw + 0) = 1;            // refcount
    *(int32_t *)(raw + 4) = 0x4C554D45;   // magic "LUME"
    char *payload = raw + 8;
    *(int32_t *)(payload) = len;           // bytes length
    return payload;
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
    char *raw = (char *)malloc(8 + total);
    *(int32_t *)(raw + 0) = 1;            // rc
    *(int32_t *)(raw + 4) = 0x4C554D45;   // magic
    ListHeader *hdr = (ListHeader *)(raw + 8);
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
    int32_t new_cap = hdr->cap * 2;
    int32_t total = (int32_t)sizeof(ListHeader) + hdr->elem_size * new_cap;
    // realloc the entire block (including rc header).
    char *raw = (char *)hdr - 8;
    raw = realloc(raw, 8 + total);
    if (!raw) {
        fprintf(stderr, "FATAL: list realloc failed (size=%d)\n", 8 + total);
        abort();
    }
    hdr = (ListHeader *)(raw + 8);
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

static GreenThread gt_pool[GT_MAX];
static int gt_count = 0;
static int gt_current = -1;
static ucontext_t gt_sched_ctx;
static int gt_event_fd = -1;

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
    char *raw = (char *)malloc(8 + 4 + len);
    *(int32_t *)(raw + 0) = 1;            // rc
    *(int32_t *)(raw + 4) = 0x4C554D45;   // magic
    char *payload = raw + 8;
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
    char *raw = (char *)malloc(8 + 4 + total);
    *(int32_t *)(raw + 0) = 1;            // rc
    *(int32_t *)(raw + 4) = 0x4C554D45;   // magic
    char *payload = raw + 8;
    *(int32_t *)payload = total;
    if (a_len > 0) memcpy(payload + 4, a + 4, a_len);
    if (b_len > 0) memcpy(payload + 4 + a_len, b + 4, b_len);
    return (int64_t)(uintptr_t)payload;
}
