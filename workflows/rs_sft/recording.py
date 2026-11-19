"""Loopback-only transparent recorder; tool implementations remain official."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import urllib.request
from urllib.parse import urlsplit
from common import MODEL, journal


def loopback(url):
    parts = urlsplit(url)
    if (parts.scheme != 'http' or parts.hostname != '127.0.0.1'
            or parts.username or parts.password or parts.query or parts.fragment):
        raise ValueError('Only plain HTTP on literal 127.0.0.1 is allowed; external APIs prohibited')
    return url.rstrip('/')


def local_json(url, payload=None):
    loopback(url)
    request = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    # Disable ambient proxy variables and redirects to remote hosts.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            raise ValueError('Redirects forbidden')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=300) as response:
        return json.load(response)


@contextmanager
def recording_proxy(endpoint, served_model, logfile):
    endpoint = loopback(endpoint)
    lock = threading.Lock()
    counter = [0]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.send_error(404)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size < 16 * 1024 * 1024:
                    raise ValueError('Invalid payload size')
                body = json.loads(self.rfile.read(size))
                if body.get('model') != MODEL or body.get('stream'):
                    raise ValueError('Unexpected model or streaming request')
                if any(k in body for k in ['models', 'route', 'provider']):
                    raise ValueError('Routing/fallback parameters forbidden')
                outgoing = dict(body, model=served_model)
                with lock:
                    counter[0] += 1
                    call_id = counter[0]
                    journal(logfile, {'type': 'request', 'call_id': call_id, 'body': outgoing})
                response = local_json(endpoint + '/chat/completions', outgoing)
                with lock:
                    journal(logfile, {'type': 'response', 'call_id': call_id, 'body': response})
                data = json.dumps(response).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except Exception as exc:
                with lock:
                    journal(logfile, {'type': 'proxy_error', 'error': repr(exc)})
                self.send_error(502, 'Local model request failed; see private experiment log')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def manager_with_journal(logfile, image):
    """Subclass transport only: all execution and isolation stay in swegemma."""
    from swegemma.sandbox.docker import ContainerManager, ContainerConfig
    class RecordedManager(ContainerManager):
        def __init__(self):
            super().__init__(ContainerConfig(image=image, network_mode='none'))
            self.record_lock = threading.RLock()

        def snapshot(self, cid):
            # Bypass the override to avoid recursive logging. Adds intent-to-add
            # for untracked files, as official submit_patch does.
            super().exec(cid, 'cd /workspace && git add -N .', timeout=30)
            result = super().exec(cid, 'cd /workspace && git diff --binary _swegemma_baseline', timeout=30)
            if result.exit_code == 0:
                journal(logfile, {'type': 'workspace_diff', 'container': cid, 'patch': result.stdout})

        def exec(self, *args, **kwargs):
            with self.record_lock:
                result = super().exec(*args, **kwargs)
                journal(logfile, {'type': 'exec', 'arguments': list(args), 'options': kwargs,
                                  'exit_code': result.exit_code, 'stdout': result.stdout,
                                  'stderr': result.stderr})
                cid = args[0] if args else kwargs.get('container_id')
                if cid:
                    self.snapshot(cid)
                return result

        def copy_to(self, *args, **kwargs):
            with self.record_lock:
                result = super().copy_to(*args, **kwargs)
                journal(logfile, {'type': 'copy_to', 'arguments': [str(x) for x in args]})
                if args and str(args[0]) in self._active_containers:
                    self.snapshot(args[0])
                return result
    return RecordedManager()
