import os, re, sys, json, shutil, shlex, subprocess
from pathlib import Path
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

OLLAMA = "http://localhost:11434"
BASE = Path(__file__).parent.resolve()
WORK = (BASE / "workspace").resolve()
WORK.mkdir(exist_ok=True)
MAX_STEPS = 10
KEEP_ALIVE = "30m"
ALLOWED = {"python", "pip", "pytest", "node", "npm", "npx", "git", "php", "composer"}

PREF = {
    "coder": ["qwen2.5-coder:32b", "qwen2.5-coder:14b", "qwen2.5-coder:7b", "deepseek-coder-v2", "codestral", "qwen2.5-coder"],
    "planner": ["deepseek-r1:8b", "deepseek-r1:7b", "deepseek-r1:14b", "deepseek-r1:32b", "llama3.1:8b"],
    "reviewer": ["deepseek-r1:8b", "deepseek-r1:7b", "deepseek-r1:14b", "deepseek-r1:32b", "llama3.1:8b"],
}

SYSTEM = """Eres un ingeniero de software senior que trabaja dentro de una carpeta de proyecto (workspace). Haces lo que el usuario pida en desarrollo de software. Respondes en espanol, directo y sin sermones.

ESTANDAR DE CODIGO
- Codigo completo y ejecutable, nunca "..." ni "resto del codigo".
- Estructura limpia: funciones pequenas, nombres claros, separacion de responsabilidades, sin codigo duplicado.
- Manejo de errores y validacion de entradas. Nada de secretos en el codigo (usa variables de entorno).
- Python: tipado, docstrings cortos, pathlib, f-strings. Web: HTML semantico y CSS responsivo.
- Si el proyecto lo amerita, crea tambien README.md, requirements.txt y tests.
- Menciona en una linea cualquier riesgo real (seguridad, rendimiento, bug).

FLUJO
1. Si no hay plan, escribe uno de 2 a 4 lineas.
2. Escribe los archivos. Para cambios pequenos en archivos existentes usa edit_file (es mas rapido que reescribir).
3. Ejecuta y prueba. Si falla, lee el error, corrige y vuelve a probar.
4. Cierra con un resumen corto: que archivos hiciste y como ejecutarlos. NO repitas el codigo en el resumen.

IMPORTANTE AL EJECUTAR: no hay teclado. Nunca uses input() ni servidores que no terminan en los comandos <run>. Para probar usa asserts, tests o argumentos. Los comandos se cortan a los 90 segundos.

HERRAMIENTAS (escribelas EXACTAMENTE asi):
<list/>
<read_file path="ruta/archivo.py"/>
<write_file path="ruta/archivo.py">
contenido completo
</write_file>
<edit_file path="ruta/archivo.py">
<old>texto exacto a reemplazar (debe aparecer una sola vez)</old>
<new>texto nuevo</new>
</edit_file>
<run>python archivo.py</run>

Rutas relativas al workspace. Puedes usar varias herramientas por mensaje y luego recibes los resultados.
Comandos permitidos en <run>: python, pip, pytest, node, npm, npx, git, php, composer."""

PLANNER = """Eres un arquitecto de software. Con el pedido del usuario y los archivos actuales, entrega un plan tecnico breve (maximo 12 lineas): stack, estructura de archivos, pasos en orden y casos borde o riesgos a cuidar. No escribas codigo. Responde en espanol."""

REVIEWER = """Eres un revisor de codigo estricto. Recibes archivos recien escritos y el pedido original. Busca bugs reales, errores de logica, fallos de seguridad (inyeccion, secretos, rutas), casos borde y cosas que no cumplen el pedido. Responde SOLO con una lista numerada de problemas concretos (archivo, funcion y como corregirlo). Si no hay problemas importantes responde exactamente: SIN_PROBLEMAS"""

PAT = re.compile(
    r'<write_file path="([^"]+)">\n?(.*?)</write_file>'
    r'|<edit_file path="([^"]+)">\s*<old>(.*?)</old>\s*<new>(.*?)</new>\s*</edit_file>'
    r'|<read_file path="([^"]+)"\s*/>'
    r'|<run>(.*?)</run>'
    r'|<list\s*/>', re.S)


def parse(text):
    acts = []
    for m in PAT.finditer(text):
        if m.group(1):
            acts.append(("write", m.group(1), m.group(2)))
        elif m.group(3):
            acts.append(("edit", m.group(3), (m.group(4), m.group(5))))
        elif m.group(6):
            acts.append(("read", m.group(6), None))
        elif m.group(7) is not None:
            acts.append(("run", m.group(7).strip(), None))
        else:
            acts.append(("list", None, None))
    return acts


def clean(c):
    c = re.sub(r'^\s*```[a-zA-Z]*\n', '', c)
    return re.sub(r'\n```\s*$', '\n', c)


def strip_think(t):
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S)
    return re.sub(r"^.*?</think>", "", t, flags=re.S).strip()


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


def t_edit(path, old, new):
    f = safe(path)
    if not f.is_file():
        return "ERROR: no existe"
    s = f.read_text(encoding="utf-8")
    old, new = old.strip("\n"), new.strip("\n")
    n = s.count(old)
    if n != 1:
        return f"ERROR: el texto a reemplazar aparece {n} veces (debe ser exactamente 1)"
    f.write_text(s.replace(old, new), encoding="utf-8")
    return f"OK: editado {path}"


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
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        r = subprocess.run(args, cwd=WORK, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=90, stdin=subprocess.DEVNULL, env=env)
        return f"codigo de salida {r.returncode}\n{(r.stdout + r.stderr)[-4000:]}"
    except subprocess.TimeoutExpired:
        return "ERROR: tiempo agotado (90s)"
    except Exception as e:
        return f"ERROR: {e}"


def installed():
    try:
        return [m["name"] for m in requests.get(f"{OLLAMA}/api/tags", timeout=5).json().get("models", [])]
    except Exception:
        return []


def pick(role, names):
    env = os.environ.get("DEV_" + role.upper()) or (os.environ.get("DEV_MODEL") if role == "coder" else None)
    if env:
        return env
    for p in PREF[role]:
        for n in names:
            if n == p or n.split(":")[0] == p:
                return n
    return None


def roles():
    names = installed()
    coder = pick("coder", names) or (names[0] if names else "qwen2.5-coder:7b")
    return {"coder": coder, "planner": pick("planner", names) or coder, "reviewer": pick("reviewer", names) or coder}


def llm(model, messages, temp=0.2, ctx=8192):
    body = {"model": model, "messages": messages, "stream": True, "keep_alive": KEEP_ALIVE,
            "options": {"temperature": temp, "num_ctx": ctx, "stop": ["Resultados de las herramientas:"]}}
    full = []
    with requests.post(f"{OLLAMA}/api/chat", json=body, stream=True, timeout=(10, 900)) as r:
        if r.status_code != 200:
            raise RuntimeError(f"Ollama {r.status_code}: {r.text[:300]}")
        for line in r.iter_lines():
            if not line:
                continue
            d = json.loads(line)
            t = d.get("message", {}).get("content", "")
            if t:
                full.append(t)
                yield ("tok", t)
            if d.get("done"):
                break
    yield ("end", "".join(full))


def ev(t, v=""):
    return json.dumps({"t": t, "v": v}) + "\n"


def run_agent(msgs, model, written):
    for _ in range(MAX_STEPS):
        yield ev("start", f"Codigo ({model})")
        text = ""
        for k, v in llm(model, msgs):
            if k == "tok":
                yield ev("tok", v)
            else:
                text = v
        msgs.append({"role": "assistant", "content": text})
        acts = parse(text)
        if not acts:
            return
        out = []
        for kind, a, b in acts:
            try:
                if kind == "write":
                    res = t_write(a, clean(b))
                elif kind == "edit":
                    res = t_edit(a, b[0], b[1])
                elif kind == "read":
                    res = t_read(a)
                elif kind == "run":
                    res = t_run(a)
                else:
                    res = t_list()
            except Exception as e:
                res = f"ERROR: {e}"
            if kind in ("write", "edit") and res.startswith("OK"):
                written.add(a)
            out.append(f"[{kind} {a or ''}]\n{res}")
        fb = "Resultados de las herramientas:\n" + "\n\n".join(out)
        msgs.append({"role": "user", "content": fb})
        yield ev("tool", fb)
    yield ev("status", "Limite de pasos alcanzado")


def review_input(message, written):
    parts = []
    for p in sorted(written):
        try:
            parts.append(f"=== {p} ===\n{t_read(p)[:5000]}")
        except Exception:
            pass
    return f"Pedido original:\n{message}\n\nArchivos:\n" + "\n\n".join(parts)[:14000]


sessions = {}


def turn(session, message, mode):
    rl = roles()
    msgs = sessions.setdefault(session, [{"role": "system", "content": SYSTEM}])
    written = set()
    content = message
    try:
        if mode == "calidad" and len(message) > 25:
            yield ev("start", f"Plan ({rl['planner']})")
            plan = ""
            pm = [{"role": "system", "content": PLANNER},
                  {"role": "user", "content": f"Archivos actuales:\n{t_list()}\n\nPedido:\n{message}"}]
            for k, v in llm(rl["planner"], pm, temp=0.3, ctx=6144):
                if k == "tok":
                    yield ev("tok", v)
                else:
                    plan = strip_think(v)
            if plan:
                content = f"{message}\n\n[Plan tecnico del arquitecto, sigue esto:]\n{plan}"
        msgs.append({"role": "user", "content": content})
        yield from run_agent(msgs, rl["coder"], written)
        if mode == "calidad" and written:
            yield ev("start", f"Revision ({rl['reviewer']})")
            review = ""
            rm = [{"role": "system", "content": REVIEWER},
                  {"role": "user", "content": review_input(message, written)}]
            for k, v in llm(rl["reviewer"], rm, temp=0.2, ctx=8192):
                if k == "tok":
                    yield ev("tok", v)
                else:
                    review = strip_think(v)
            if review and "SIN_PROBLEMAS" not in review:
                msgs.append({"role": "user", "content": "Un revisor encontro estos problemas. Corrige los que sean reales, prueba de nuevo y cierra con un resumen corto:\n" + review})
                yield from run_agent(msgs, rl["coder"], written)
    except Exception as e:
        yield ev("error", str(e))
    if len(msgs) > 40:
        sessions[session] = msgs[:1] + msgs[-30:]
    yield ev("done")


class Req(BaseModel):
    session: str = "default"
    message: str = ""
    mode: str = "rapido"


app = FastAPI()


@app.post("/api/chat")
def chat(req: Req):
    return StreamingResponse(turn(req.session, req.message, req.mode), media_type="application/x-ndjson")


@app.post("/api/reset")
def reset(req: Req):
    sessions.pop(req.session, None)
    return {"ok": True}


@app.get("/api/files")
def files():
    return {"roles": roles(), "files": t_list().split("\n")}


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE / "index.html").read_text(encoding="utf-8")