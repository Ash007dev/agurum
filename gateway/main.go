package main

import (
	"bufio"
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"
)

func main() {
	capStr := os.Getenv("RING_BUFFER_CAP")
	capVal := 10000
	if capStr != "" {
		if val, err := strconv.Atoi(capStr); err == nil {
			capVal = val
		}
	}

	batchSizeStr := os.Getenv("BATCH_SIZE")
	batchSize := 100
	if batchSizeStr != "" {
		if val, err := strconv.Atoi(batchSizeStr); err == nil {
			batchSize = val
		}
	}

	intervalStr := os.Getenv("BATCH_INTERVAL_MS")
	intervalMs := 100
	if intervalStr != "" {
		if val, err := strconv.Atoi(intervalStr); err == nil {
			intervalMs = val
		}
	}

	rb := NewRingBuffer(capVal)
	
	targetURL := "http://unix/batch"
	if os.Getenv("PCE_UDS_PATH") == "" {
		targetURL = "http://127.0.0.1:8000/batch"
	}
	f := NewFlusher(rb, targetURL, batchSize, intervalMs)

	ctx, cancel := context.WithCancel(context.Background())

	go f.Start(ctx)

	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		status := map[string]interface{}{
			"status":      "ok",
			"buffer_size": rb.Size(),
			"drops_total": rb.Drops(),
		}
		json.NewEncoder(w).Encode(status)
	})

	http.HandleFunc("/inject", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var events []json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&events); err != nil {
			http.Error(w, "Invalid JSON array", http.StatusBadRequest)
			return
		}
		for _, ev := range events {
			if ValidateEvent(ev) {
				rb.Push(ev)
			}
		}
		w.WriteHeader(http.StatusAccepted)
	})

	port := os.Getenv("GATEWAY_PORT")
	if port == "" {
		port = "8080"
	}
	srv := &http.Server{Addr: ":" + port}
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("HTTP server error: %v", err)
		}
	}()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	stdinDone := make(chan struct{})
	go func() {
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}

			// Must copy to avoid underlying byte array overwrite by scan
			msg := make(json.RawMessage, len(line))
			copy(msg, line)

			if ValidateEvent(msg) {
				rb.Push(msg)
			}
		}
		close(stdinDone)
	}()

	select {
	case <-sigChan:
		log.Println("Received termination signal")
	case <-stdinDone:
		log.Println("Stdin closed")
	}

	log.Println("Shutting down...")

	// Grace shutdown
	timeoutCtx, cancelTimeout := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancelTimeout()
	srv.Shutdown(timeoutCtx)

	cancel()                           // signal flusher to drain whatever's left
	time.Sleep(500 * time.Millisecond) // Let flusher finish the last post
	log.Println("Exiting")
}
