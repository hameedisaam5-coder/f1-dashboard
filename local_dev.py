"""
Local dev server for F1 Dashboard.
Serves static HTML/CSS/JS files and routes /api/* calls to their Python handlers.
Usage: python local_dev.py
"""
import sys
import os
import io
import json

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import mimetypes

PORT = 3000
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

class DevHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/')

        # Route /api/* calls to Python handlers
        if path == '/api/replay_data':
            self._route_api('replay_data', parsed.query)
        elif path == '/api/race_data':
            self._route_api('race_data', parsed.query)
        elif path == '' or path == '/':
            self._serve_file('/dashboard.html')
        else:
            self._serve_file(path)

    def _route_api(self, module_name, query_string):
        try:
            # Dynamically import the handler
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                module_name, 
                os.path.join(PROJECT_ROOT, 'api', f'{module_name}.py')
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Call build function directly based on module
            qs = parse_qs(query_string)
            if module_name == 'replay_data':
                year = int(qs.get('year', ['2025'])[0])
                race = qs.get('race', ['Australian Grand Prix'])[0]
                session = qs.get('session', ['R'])[0].upper()
                data = mod.build_replay_data(year, race, session)
                body = json.dumps(data, allow_nan=False).replace('NaN', 'null').encode()
            elif module_name == 'race_data':
                # Fallback for race_data
                year = int(qs.get('year', ['2025'])[0])
                race = qs.get('race', ['Australian Grand Prix'])[0]
                session = qs.get('session', ['R'])[0].upper()
                data = mod.build_race_data(year, race, session) if hasattr(mod, 'build_race_data') else {}
                body = json.dumps(data).encode()
            else:
                body = b'{"error":"unknown api"}'
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            import traceback
            traceback.print_exc()
            body = json.dumps({'error': str(e)}).encode()
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)

    def _serve_file(self, path):
        file_path = os.path.join(PROJECT_ROOT, path.lstrip('/'))
        if not os.path.exists(file_path):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
            return
        
        content_type, _ = mimetypes.guess_type(file_path)
        content_type = content_type or 'text/html'
        
        with open(file_path, 'rb') as f:
            body = f.read()
        
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[DEV] {self.command} {self.path} → {args[1] if len(args) > 1 else ''}")

if __name__ == '__main__':
    print(f"🏎️  F1 Dashboard Dev Server running on http://localhost:{PORT}")
    print(f"   Open: http://localhost:{PORT}/replay.html")
    print(f"   Press Ctrl+C to stop")
    server = HTTPServer(('', PORT), DevHandler)
    server.serve_forever()
