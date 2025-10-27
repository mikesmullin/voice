"""Voice synthesis client for hot reloading."""

import socket
import json
import sys


def send_synthesis_request(
    voice_name: str,
    text: str,
    output_file: str = None,
    host: str = "127.0.0.1",
    port: int = 3124,
    connection_timeout: float = 0.5
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
    
    Returns:
        Response dict with 'status' or 'error' key, or None if connection failed
    """
    
    # Create request
    request = {
        "voice": voice_name,
        "text": text,
        "output_file": output_file
    }
    
    try:
        # Connect to server with short timeout
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(connection_timeout)
        client_socket.connect((host, port))
        
        # Once connected, increase timeout for actual synthesis
        client_socket.settimeout(30.0)
        
        # Send request
        request_data = json.dumps(request).encode('utf-8') + b'\n'
        client_socket.sendall(request_data)
        
        # Receive response
        response_data = b""
        while True:
            chunk = client_socket.recv(4096)
            if not chunk:
                break
            response_data += chunk
            if b"\n" in response_data:
                break
        
        client_socket.close()
        
        if not response_data:
            return {"error": "No response from server"}
        
        response = json.loads(response_data.decode('utf-8'))
        return response
        
    except (ConnectionRefusedError, socket.timeout, OSError):
        # Return None to signal fallback to direct synthesis
        return None
    except Exception as e:
        return {"error": str(e)}
