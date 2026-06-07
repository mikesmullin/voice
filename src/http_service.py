"""HTTP speech service for browser-compatible audio streaming."""

from __future__ import annotations

import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .timing import log, start_timer
from .voice_engine import VoiceEngine


class HttpVoiceService:
    """Simple HTTP service exposing a speech endpoint."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 3040,
        config_path: Optional[str] = None,
        force_cpu: bool = False,
    ):
        self.host = host
        self.port = port
        self.engine = VoiceEngine(config_path=config_path, force_cpu=force_cpu)

    def start(self) -> None:
        """Start the HTTP service and block until interrupted."""
        start_timer()
        log("[Service] Initializing HTTP speech service...")

        service = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _resolve_voice(self, engine_name: str, requested_voice: Optional[str]) -> str:
                default_by_engine = {
                    "piper": "lessac",
                    "kokoro": "bella",
                }

                if requested_voice:
                    info = service.engine.get_voice_info(requested_voice)
                    configured_engine = info.get("engine", "kokoro")
                    if configured_engine != engine_name:
                        raise ValueError(
                            f"Voice '{requested_voice}' uses engine '{configured_engine}', "
                            f"but engine='{engine_name}' was requested"
                        )
                    return requested_voice

                default_voice = default_by_engine.get(engine_name)
                if default_voice:
                    info = service.engine.get_voice_info(default_voice)
                    if info.get("engine", "kokoro") == engine_name:
                        return default_voice

                # Fallback: first configured voice for the requested engine.
                for preset in service.engine.list_voices():
                    info = service.engine.get_voice_info(preset)
                    if info.get("engine", "kokoro") == engine_name:
                        return preset

                raise ValueError(f"No configured voice presets available for engine '{engine_name}'")

            def do_GET(self) -> None:
                parsed = urlparse(self.path)

                if parsed.path == "/health":
                    self._send_json(200, {"status": "ok"})
                    return

                if parsed.path != "/speak":
                    self._send_json(
                        404,
                        {
                            "error": "Not found",
                            "usage": "/speak?q=<text>&engine=piper&voice=lessac&gain=1.0",
                        },
                    )
                    return

                try:
                    query = parse_qs(parsed.query, keep_blank_values=False)
                    text = (query.get("q", [""])[0] or "").strip()
                    engine_name = (query.get("engine", ["piper"])[0] or "piper").strip().lower()
                    requested_voice = (query.get("voice", [""])[0] or "").strip() or None
                    gain_str = (query.get("gain", ["1.0"])[0] or "1.0").strip()

                    if not text:
                        self._send_json(400, {"error": "Missing required query parameter: q"})
                        return

                    if engine_name not in {"piper", "kokoro"}:
                        self._send_json(400, {"error": "Invalid engine. Supported values: piper, kokoro"})
                        return

                    gain = float(gain_str)
                    if gain < 0:
                        self._send_json(400, {"error": "gain must be greater than or equal to 0"})
                        return

                    voice_name = self._resolve_voice(engine_name, requested_voice)

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        wav_path = tmp.name

                    try:
                        service.engine.synthesize(
                            text=text,
                            voice_name=voice_name,
                            output_file=wav_path,
                            gain=gain,
                        )
                        with open(wav_path, "rb") as wav_file:
                            wav_bytes = wav_file.read()
                    finally:
                        try:
                            os.unlink(wav_path)
                        except FileNotFoundError:
                            pass

                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(wav_bytes)))
                    self.send_header("Content-Disposition", 'inline; filename="speech.wav"')
                    self.end_headers()
                    self.wfile.write(wav_bytes)

                    log(
                        f"[Service] Served /speak voice='{voice_name}' engine='{engine_name}' "
                        f"chars={len(text)} bytes={len(wav_bytes)}"
                    )

                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
                except Exception as exc:
                    log(f"[Service] Error handling request: {exc}")
                    self._send_json(500, {"error": str(exc)})

            def log_message(self, format: str, *args) -> None:
                log(f"[HTTP] {self.address_string()} - {format % args}")

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        log(f"[Service] Listening on http://{self.host}:{self.port}")
        print(f"HTTP speech service ready at http://{self.host}:{self.port}")
        print("Try: /speak?q=Hello&engine=piper&voice=lessac")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            log("[Service] Shutting down...")
        finally:
            server.server_close()
            log("[Service] Stopped")


def start_http_service(
    config_path: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 3040,
    force_cpu: bool = False,
) -> None:
    """Start the HTTP speech service."""
    service = HttpVoiceService(host=host, port=port, config_path=config_path, force_cpu=force_cpu)
    service.start()
