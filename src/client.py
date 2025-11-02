"""Voice synthesis client for hot reloading."""

import socket
import json
import sys
import signal


def send_synthesis_request(
    voice_name: str,
    text: str,
    output_file: str = None,
    host: str = "127.0.0.1",
    port: int = 3124,
    connection_timeout: float = 0.5,
    stinger: str = None
) -> dict:
    """
    Send a synthesis request to the voice server.
    
    Args:
        voice_name: Voice preset name
        text: Text to synthesize
        output_file: Optional output file path
        host: Server host
        port: Server port
        connection_timeout: Connection timeout in seconds (default: 0.5)
        stinger: Optional stinger name to play before speech
    
    Returns:
        Response dict with 'status' or 'error' key, or None if connection failed
    """
    
    # Create request
    request = {
        "voice": voice_name,
        "text": text,
        "output_file": output_file,
        "stinger": stinger
    }
    
    client_socket = None
    interrupted = [False]  # Use list to allow modification in nested function
    
    def cleanup_and_exit(signum, frame):
        """Handle CTRL+C by closing socket cleanly."""
        print("\n[Client] Interrupted by user")
        interrupted[0] = True
        if client_socket:
            try:
                client_socket.shutdown(socket.SHUT_RDWR)
                client_socket.close()
            except:
                pass
        raise KeyboardInterrupt()
    
    # Install signal handler
    old_handler = signal.signal(signal.SIGINT, cleanup_and_exit)
    
    try:
        # Connect to server with short timeout
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(connection_timeout)
        client_socket.connect((host, port))
        
        # Once connected, set a reasonable timeout for recv operations
        # This allows CTRL+C to be checked periodically
        client_socket.settimeout(1.0)
        
        # Send request
        request_data = json.dumps(request).encode('utf-8') + b'\n'
        client_socket.sendall(request_data)
        
        # Receive response with periodic timeout checks
        response_data = b""
        timeout_count = 0
        max_timeouts = 600  # 10 minutes at 1 second intervals
        
        while not interrupted[0]:
            try:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                if b"\n" in response_data:
                    break
                timeout_count = 0  # Reset timeout counter on successful recv
            except socket.timeout:
                # Check if we've waited too long
                timeout_count += 1
                if timeout_count >= max_timeouts:
                    return {"error": "Server timeout"}
                # Continue waiting
                continue
        
        if interrupted[0]:
            return {"error": "Interrupted by user"}
        
        client_socket.close()
        
        if not response_data:
            return {"error": "No response from server"}
        
        response = json.loads(response_data.decode('utf-8'))
        return response
        
    except KeyboardInterrupt:
        # User pressed CTRL+C
        return {"error": "Interrupted by user"}
    except (ConnectionRefusedError, socket.timeout, OSError):
        # Return None to signal fallback to direct synthesis
        return None
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, old_handler)


