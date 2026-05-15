"""
Shared UDS mock-server helper used by multiple test modules.
Runs a minimal HTTP/1.1 server over a Unix-domain socket in a background
thread.  The server accepts connections, reads one request, replies 200 OK,
and loops.  It does NOT need to be a full HTTP parser — the Go flusher only
checks the status code.
"""
import os
import socket
import threading


def run_uds_server(sock_path: str) -> socket.socket:
    """
    Bind a Unix-domain socket at *sock_path*, start a background accept-loop,
    and return the server socket (caller is responsible for closing it).
    """
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(sock_path)
    server.listen(32)

    def _accept_loop():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                # server was closed — exit cleanly
                break
            try:
                # Drain the request (up to 64 KiB) so the sender doesn't block
                data = b""
                conn.settimeout(2.0)
                try:
                    while True:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                        # A complete HTTP request ends with double CRLF; after
                        # that comes the body.  Parse Content-Length to know
                        # when we have everything.
                        if b"\r\n\r\n" in data:
                            header_part = data.split(b"\r\n\r\n", 1)[0]
                            body_part   = data.split(b"\r\n\r\n", 1)[1]
                            cl = 0
                            for line in header_part.split(b"\r\n"):
                                if line.lower().startswith(b"content-length:"):
                                    cl = int(line.split(b":", 1)[1].strip())
                            if len(body_part) >= cl:
                                break
                except (socket.timeout, OSError):
                    pass
                # Respond 200 OK
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Length: 0\r\n"
                    b"Connection: close\r\n"
                    b"\r\n"
                )
            except Exception:
                pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    t = threading.Thread(target=_accept_loop, daemon=True)
    t.start()
    return server
