import os, re, sys, shutil, shlex, subprocess
from pathlib import Path
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

OLLAMA = "http://localhost:11434"
MODEL = os.environ.get("DEV_MODEL", "qwen2.5-coder:7b")
BASE = Path(__file__).parent.resolve()
WORK = (BASE / "workspace").resolve()
WORK.mkdir(exist_ok=True)
MAX_STEPS = 12
ALLOWED = {"python", "pip", "pytest", "node", "npm", "npx", "git", "php", "composer"}

SYSTEM = """Eres un ingeniero de software senior que trabaja dentro de una carpeta de proyecto (workspace).
Cumples lo que el usuario pida en desarrollo de software. Responde en espanol, directo y sin sermones.
Antes de programar, da un plan de 2 a 4 lineas. Entrega codigo completo, sin omitir partes.
Si ves un riesgo real (seguridad, bug), mencionalo en una linea.

Para actuar sobre el proyecto usa estas herramientas, escritas EXACTAMENTE asi:
<list/>
<read_file path="ruta/archivo.py"/>
<write_file path="ruta/archivo.py">
contenido completo del archivo
</write_file>
<run>python archivo.py</run>

Reglas: las rutas son relativas al workspace. Puedes usar varias herramientas en un mensaje y luego recibiras los resultados.
Si un comando falla, lee el error, corrige el codigo y vuelve a probar.
Cuando termines, responde con un resumen SIN usar herramientas.
Comandos permitidos en <run>: python, pip, pytest, node, npm, npx, git, php, composer."""

PAT = re.compile(r'<write_file path="([^"]+)">\n?(.*?)</write_file>|<read_file path="([^"]+)"\s*/>|<run>(.*?)</run>|<list\s*/>', re.S)


def parse(text):
    acts = []
    for m in PAT.finditer(text):
        if m.group(1):
            acts.append(("write", m.group(1), m.group(2)))
        elif m.group(3):
            acts.append(("read", m.group(3), None))
        elif m.group(4) is not None:
            acts.append(("run", m.group(4).strip(), None))
        else:
            acts.append(("list", None, None))
    return acts


def clean(c):
    c = re.sub(r'^\s*```[a-zA-Z]*\n', '', c)
    return re.sub(r'\n```\s*$', '\n', c)


def safe(p):
    full = (WORK / p).resolve()
    full.relative_to(WORK)
    return full


def t_list():
    skip = {"venv", ".venv", "node_modules", "__pycache__", ".git"}
    items = [str(f.relative_to(WORK)) for f in WORK.rglob("*") if f.is_file() and not (skip & set(f.parts))]
    return "\n".join(sorted(items)[:200]) or "(vacio)"


def t_read(path):
    f = safe(path)
    return f.read_text(encoding="utf-8", errors="replace")[:8000] if f.is_file() else "ERROR: no existe"


def t_write(path, content):
    f = safe(path)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content, encoding="utf-8")
    return f"OK: escrito {path} ({len(content)} caracteres)"


def t_run(cmd):
    try:
        args = shlex.split(cmd, posix=False)
    except ValueError as e:
        return f"ERROR: {e}"
    if not args:
        return "ERROR: comando vacio"
    name = Path(args[0]).stem.lower()
    if name not in ALLOWED:
        return f"ERROR: comando no permitido ({name})"
    if name == "python":
        args[0] = sys.executable
    elif name == "pip":
        args = [sys.executable, "-m", "pip"] + args[1:]
    elif name == "pytest":
        args = [sys.executable, "-m", "pytest"] + args[1:]
    else:
        exe = shutil.which(args[0])
        if not exe:
            return f"ERROR: {name} no esta instalado"
        args[0] = exe
    try:
        r = subprocess.run(args, cwd=WORK, capture_output=True, text=True, errors="replace", timeout=90)
        return f"codigo de salida {r.returncode}\n{(r.stdout + r.stderr)[-4000:]}"
    except subprocess.TimeoutExpired:
        return "ERROR: tiempo agotado (90s)"
    except Exception as e:
        return f"ERROR: {e}"


def call_model(messages):
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": MODEL, "messages": messages, "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192}}, timeout=900)
    r.raise_for_status()
    return r.json()["message"]["content"]


class Req(BaseModel):
    session: str = "default"
    message: str = ""


sessions = {}
app = FastAPI()


@app.post("/api/chat")
def chat(req: Req):
    msgs = sessions.setdefault(req.session, [{"role": "system", "content": SYSTEM}])
    msgs.append({"role": "user", "content": req.message})
    log = []
    try:
        for _ in range(MAX_STEPS):
            text = call_model(msgs)
            msgs.append({"role": "assistant", "content": text})
            log.append({"role": "assistant", "content": text})
            acts = parse(text)
            if not acts:
                break
            out = []
            for kind, a, b in acts:
                try:
                    if kind == "write":
                        res = t_write(a, clean(b))
                    elif kind == "read":
                        res = t_read(a)
                    elif kind == "run":
                        res = t_run(a)
                    else:
                        res = t_list()
                except Exception as e:
                    res = f"ERROR: {e}"
                out.append(f"[{kind} {a or ''}]\n{res}")
            fb = "Resultados de las herramientas:\n" + "\n\n".join(out)
            msgs.append({"role": "user", "content": fb})
            log.append({"role": "tool", "content": fb})
    except Exception as e:
        log.append({"role": "error", "content": str(e)})
    if len(msgs) > 40:
        sessions[req.session] = msgs[:1] + msgs[-30:]
    return {"log": log}


@app.post("/api/reset")
def reset(req: Req):
    sessions.pop(req.session, None)
    return {"ok": True}


@app.get("/api/files")
def files():
    return {"model": MODEL, "files": t_list().split("\n")}


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE / "index.html").read_text(encoding="utf-8")