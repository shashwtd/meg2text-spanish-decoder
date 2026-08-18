"""Minimal local web interface."""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from .predict import PACKAGE_ROOT, Predictor


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spanish MEG decoder</title><style>
*{box-sizing:border-box}body{margin:0;background:#fff;color:#111;font:16px Arial,sans-serif;line-height:1.45}
main{width:min(720px,calc(100% - 32px));margin:48px auto}h1{font-size:28px;margin:0 0 8px}p{color:#444;margin:0 0 24px}
label{display:block;font-weight:600;margin:18px 0 8px}select,input,button{width:100%;padding:12px;border:1px solid #999;background:#fff;font:inherit}
button{margin-top:12px;background:#111;color:#fff;border-color:#111;font-weight:600;cursor:pointer}button:disabled{background:#777}
#status{margin:18px 0;color:#444}#error{color:#9b1c1c}.result{display:none;border-top:1px solid #aaa;margin-top:28px;padding-top:22px}
.prediction{font-size:26px;font-weight:600;margin:8px 0 20px}.target{padding:12px;border:1px solid #bbb;margin:8px 0}.meta{color:#555;margin-top:16px}
</style></head><body><main><h1>Spanish MEG decoder</h1><p>Choose a held-out MEG example or upload a compatible NPZ file.</p>
<label for="example">Included example</label><select id="example"></select><button id="decode">Decode example</button>
<label for="file">Or upload NPZ</label><input id="file" type="file" accept=".npz"><button id="upload">Decode upload</button>
<div id="status"></div><div id="error"></div><section id="result" class="result"><div>Predicted text</div><div id="prediction" class="prediction"></div>
<div>Recorded typed text</div><div id="target" class="target"></div><div id="cer"></div><div id="meta" class="meta"></div></section>
<script>
const q=id=>document.getElementById(id);let examples=[];
fetch('/api/examples').then(r=>r.json()).then(d=>{examples=d;q('example').innerHTML=d.map(x=>`<option value="${x.file}">${x.file} — subject ${x.subject}</option>`).join('')});
function show(d){q('prediction').textContent=d.prediction||'(empty)';q('target').textContent=d.target_text||'Not included';q('cer').textContent=d.cer===undefined?'':`CER for this example: ${(100*d.cer).toFixed(1)}%`;q('meta').textContent=`${d.duration_seconds.toFixed(2)} seconds of MEG · ${d.inference_seconds.toFixed(2)} seconds on ${d.device}`;q('result').style.display='block'}
async function run(url,body,button){button.disabled=true;q('status').textContent='Decoding…';q('error').textContent='';q('result').style.display='none';try{const r=await fetch(url,{method:'POST',body});const d=await r.json();if(!r.ok)throw new Error(d.error||'Prediction failed');show(d);q('status').textContent=''}catch(e){q('error').textContent=e.message;q('status').textContent=''}finally{button.disabled=false}}
q('decode').onclick=()=>run('/api/example/'+encodeURIComponent(q('example').value),null,q('decode'));
q('upload').onclick=()=>{const f=q('file').files[0];if(f)run('/api/predict',f,q('upload'))};
</script></main></body></html>"""


class Handler(BaseHTTPRequestHandler):
    predictor: Predictor
    examples_dir = PACKAGE_ROOT / "test_cases" / "inputs"
    lock = threading.Lock()

    def send_content(self, content: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, value: object, status: int = 200) -> None:
        self.send_content(
            json.dumps(value, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_content(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/examples":
            rows = []
            for path in sorted(self.examples_dir.glob("*.npz")):
                try:
                    from .predict import load_npz

                    sample = load_npz(path)
                    rows.append({"file": path.name, "subject": sample["subject"]})
                except Exception:
                    continue
            self.send_json(rows)
        elif self.path == "/health":
            self.send_json({"status": "ok"})
        else:
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            if self.path.startswith("/api/example/"):
                name = Path(unquote(self.path.removeprefix("/api/example/"))).name
                source: object = self.examples_dir / name
                if not Path(source).is_file():
                    raise ValueError("Example not found")
            elif self.path == "/api/predict":
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 100 * 1024 * 1024:
                    raise ValueError("Choose an NPZ smaller than 100 MB")
                source = self.rfile.read(length)
            else:
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            with self.lock:
                result = self.predictor.predict(source)
            self.send_json(result)
        except (ValueError, OSError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self.send_json({"error": f"Prediction failed: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Spanish MEG decoder UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    Handler.predictor = Predictor(device=args.device)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Spanish MEG decoder: {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

