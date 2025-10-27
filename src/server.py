"""Voice synthesis server for low-latency hot reloading."""

import socket
import json
import threading
import sys
from typing import Optional

from .voice_engine import VoiceEngine
from .timing import start_timer, log


class VoiceServer:
    """TCP server for low-latency voice synthesis."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 3124, config_path: Optional[str] = None, force_cpu: bool = False):
        self.host = host
        self.port = port
        self.engine = VoiceEngine(config_path=config_path, force_cpu=force_cpu)
        self.running = False
        self.socket = None
        
    def start(self):
        """Start the voice synthesis server."""
        start_timer()
        log("[Server] Initializing voice engine...")
        
        # Pre-initialize the pipeline to load model onto GPU
        self.engine._initialize_pipeline()
        log("[Server] Model loaded and ready")
        
        # Create TCP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        
        self.running = True
        log(f"[Server] Listening on {self.host}:{self.port}")
        print(f"Voice server ready. Use 'voice hot <preset> <text>' to synthesize.")
        
        try:
            while self.running:
                try:
                    # Accept connection with timeout to allow clean shutdown
                    self.socket.settimeout(1.0)
                    try:
                        client_socket, address = self.socket.accept()
                    except socket.timeout:
                        continue
                    
                    # Handle request in a separate thread
                    thread = threading.Thread(target=self._handle_client, args=(client_socket,))
                    thread.daemon = True
                    thread.start()
                    
                except KeyboardInterrupt:
                    log("[Server] Shutting down...")
                    break
                except Exception as e:
                    log(f"[Server] Error: {e}")
                    
        finally:
            self.running = False
            if self.socket:
                self.socket.close()
            log("[Server] Stopped")
    
    def _handle_client(self, client_socket: socket.socket):
        """Handle a client request."""
        try:
            # Receive data
            data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break
            
            if not data:
                return
            
            # Parse request
            request = json.loads(data.decode('utf-8'))
            
            voice_name = request.get("voice")
            text = request.get("text")
            output_file = request.get("output_file")
            
            if not voice_name or not text:
                client_socket.sendall(b'{"error": "Missing voice or text"}\n')
                return
            
            start_timer()
            log(f"[Server] Request: voice='{voice_name}', text='{text[:50]}...'")
            
            # Synthesize speech
            try:
                self.engine.synthesize(text, voice_name, output_file)
                response = {"status": "success"}
            except Exception as e:
                log(f"[Server] Synthesis error: {e}")
                response = {"error": str(e)}
            
            # Send response
            client_socket.sendall(json.dumps(response).encode('utf-8') + b'\n')
            
        except Exception as e:
            log(f"[Server] Client handler error: {e}")
            try:
                client_socket.sendall(json.dumps({"error": str(e)}).encode('utf-8') + b'\n')
            except:
                pass
        finally:
            client_socket.close()


def start_server(config_path: Optional[str] = None, host: str = "127.0.0.1", port: int = 3124, force_cpu: bool = False):
    """Start the voice server."""
    server = VoiceServer(host=host, port=port, config_path=config_path, force_cpu=force_cpu)
    server.start()
