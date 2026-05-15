package main

import (
	"bytes"
	"context"
	"encoding/json"
	"log"
	"net"
	"net/http"
	"os"
	"time"
)

type Flusher struct {
	rb         *RingBuffer
	targetURL  string
	client     *http.Client
	batchSize  int
	intervalMs int
}

func NewFlusher(rb *RingBuffer, targetURL string, batchSize int, intervalMs int) *Flusher {
	udsPath := os.Getenv("PCE_UDS_PATH")
	
	dialer := &net.Dialer{
		Timeout: 5 * time.Second,
	}

	transport := &http.Transport{
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			if udsPath != "" {
				return dialer.DialContext(ctx, "unix", udsPath)
			}
			// Fallback to TCP for Windows
			return dialer.DialContext(ctx, "tcp", "127.0.0.1:8000")
		},
	}

	client := &http.Client{
		Transport: transport,
		Timeout:   10 * time.Second,
	}

	return &Flusher{
		rb:         rb,
		targetURL:  targetURL,
		client:     client,
		batchSize:  batchSize,
		intervalMs: intervalMs,
	}
}

func (f *Flusher) Start(ctx context.Context) {
	ticker := time.NewTicker(time.Duration(f.intervalMs) * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			f.flushAndSend()
			return
		case <-ticker.C:
			f.flushAndSend()
		default:
			if f.rb.Size() >= f.batchSize {
				f.flushAndSend()
			} else {
				time.Sleep(5 * time.Millisecond)
			}
		}
	}
}

func (f *Flusher) flushAndSend() {
	items := f.rb.PopBatch(f.batchSize)
	if len(items) == 0 {
		return
	}

	data, err := json.Marshal(items)
	if err != nil {
		log.Printf("Flusher encode error: %v", err)
		return
	}

	req, err := http.NewRequest("POST", f.targetURL, bytes.NewReader(data))
	if err != nil {
		log.Printf("Flusher req create error: %v", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := f.client.Do(req)
	if err != nil {
		log.Printf("Flusher HTTP POST error: %v, dropped %d events", err, len(items))
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		log.Printf("Flusher returned status %d, dropped %d events", resp.StatusCode, len(items))
	}
}
