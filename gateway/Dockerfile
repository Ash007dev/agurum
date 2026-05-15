# ── Build stage ───────────────────────────────────────────────────────────────
FROM golang:1.22-alpine AS builder

WORKDIR /app

# Copy module files first for layer caching
COPY go.mod ./
RUN go mod download

# Copy source and build a static binary
COPY *.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o gateway .

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM alpine:3.19

# ca-certificates needed for any outbound TLS; tini for proper signal handling
RUN apk add --no-cache ca-certificates tini

WORKDIR /app
COPY --from=builder /app/gateway .

# Environment variable defaults (all overridable at runtime)
ENV RING_BUFFER_CAP=10000 \
    BATCH_SIZE=100 \
    BATCH_INTERVAL_MS=100 \
    UDS_PATH=/tmp/pce.sock \
    GATEWAY_PORT=8080

# Expose HTTP health/inject port
EXPOSE 8080

# Use tini as PID 1 so SIGTERM is forwarded correctly to the gateway process
ENTRYPOINT ["/sbin/tini", "--"]
CMD ["./gateway"]
