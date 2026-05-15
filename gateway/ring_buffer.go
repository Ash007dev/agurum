package main

import (
	"encoding/json"
	"sync"
)

type RingBuffer struct {
	mu    sync.Mutex
	buf   []json.RawMessage
	head  int
	tail  int
	size  int
	cap   int
	drops int
}

func NewRingBuffer(capacity int) *RingBuffer {
	return &RingBuffer{
		buf: make([]json.RawMessage, capacity),
		cap: capacity,
	}
}

func (r *RingBuffer) Push(msg json.RawMessage) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.size == r.cap {
		r.drops++
		return false
	}

	r.buf[r.tail] = msg
	r.tail = (r.tail + 1) % r.cap
	r.size++
	return true
}

func (r *RingBuffer) PopBatch(batchSize int) []json.RawMessage {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.size == 0 {
		return nil
	}

	limit := batchSize
	if r.size < batchSize {
		limit = r.size
	}

	res := make([]json.RawMessage, 0, limit)
	for i := 0; i < limit; i++ {
		res = append(res, r.buf[r.head])
		r.head = (r.head + 1) % r.cap
	}

	r.size -= limit
	return res
}

func (r *RingBuffer) Flush() []json.RawMessage {
	return r.PopBatch(r.cap)
}

func (r *RingBuffer) Size() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.size
}

func (r *RingBuffer) Drops() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.drops
}
