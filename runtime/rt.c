// Lumen actor runtime: single-threaded event loop with message queue.
// Compiled to rt.o and linked with Lumen programs.

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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
    queue[queue_tail].target_cell = cell;
    queue[queue_tail].dispatch = dispatch;
    queue[queue_tail].msg_kind = kind;
    queue[queue_tail].arg0 = arg0;
    queue[queue_tail].reply_slot = NULL;
    queue_tail = (queue_tail + 1) % QUEUE_CAP;
}

// Push a message and get a reply (blocks by draining until the reply
// is filled). For the single-threaded event loop, "blocking" means
// processing other messages until ours is handled.
int64_t lumen_rt_ask(void *cell,
                     void (*dispatch)(void *, int32_t, int64_t, int64_t *),
                     int32_t kind, int64_t arg0) {
    int64_t reply = 0;
    // Enqueue with a reply slot.
    queue[queue_tail].target_cell = cell;
    queue[queue_tail].dispatch = dispatch;
    queue[queue_tail].msg_kind = kind;
    queue[queue_tail].arg0 = arg0;
    queue[queue_tail].reply_slot = &reply;
    queue_tail = (queue_tail + 1) % QUEUE_CAP;
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
