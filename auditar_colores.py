"""Auditoría de colores de docs/index.html: dónde se usa cada hex y si respeta un color = un rol.

    python auditar_colores.py ruta/a/docs/index.html
"""
import itertools
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROLES = {
    "rareza": ["#9fb3cc", "#5e98d9", "#4b69ff", "#8847ff", "#d32ce6", "#eb4b4b"],
    "modelo": ["#e8962e", "#f0a93b"],
    "origen": ["#26b5c7", "#9a724a", "#cea8d5", "#6f6a64"],
    "armory": ["#ffab91"],
    "gris": ["#919191"],
}
ROL_DE = {c: r for r, cs in ROLES.items() for c in cs}
NOMBRE = dict(zip(ROLES["rareza"], ["Consumer", "Industrial", "Mil-Spec", "Restricted", "Classified", "Covert"]))
NOMBRE.update(zip(ROLES["origen"], ["caja", "drop de mapa", "paquete souvenir", "sin identificar"]))
NOMBRE.update({"#e8962e": "modelo (ámbar)", "#f0a93b": "--accent", "#ffab91": "Armory", "#919191": "gris"})
HEX = re.compile(r"#[0-9a-fA-F]{6}\b")

html = Path(sys.argv[1]).read_text(encoding="utf-8")
m = re.search(r"const APP = (\{.*?\});\n", html, flags=re.S)
APP = json.loads(m.group(1).replace("<\\/", "</"))
resto = html[:m.start()] + html[m.end():]

usos = defaultdict(list)          # hex → [(dónde, qué)]

# 1 · ESPECS: cada hex con su gráfico y, si está en una escala, la categoría que pinta
def recorrer(nodo, grafico, ruta):
    if isinstance(nodo, dict):
        if isinstance(nodo.get("range"), list):
            dom = nodo.get("domain") if isinstance(nodo.get("domain"), list) else [None] * len(nodo["range"])
            for d, c in zip(dom, nodo["range"]):
                if isinstance(c, str) and HEX.fullmatch(c):
                    usos[c.lower()].append((f"ESPECS.{grafico}", f"escala: «{d}»"))
        for k, v in nodo.items():
            if k == "range" and isinstance(v, list):
                continue
            if isinstance(v, str) and HEX.fullmatch(v):
                usos[v.lower()].append((f"ESPECS.{grafico}", f"{ruta}.{k}"))
            else:
                recorrer(v, grafico, f"{ruta}.{k}")
    elif isinstance(nodo, list):
        for i, v in enumerate(nodo):
            recorrer(v, grafico, f"{ruta}[{i}]")


for g, spec in APP["ESPECS"].items():
    spec = {k: v for k, v in spec.items() if k != "datasets"}
    recorrer(spec, g, "")
for clave in ("COL_RAREZA", "COL_ORIGEN"):
    for i, c in enumerate(APP[clave]):
        usos[c.lower()].append((f"APP.{clave}", f"[{i}]"))

# 2 · CSS y HTML generados (todo lo que no es el objeto APP)
for n, linea in enumerate(resto.splitlines(), start=1):
    for h in HEX.findall(linea):
        trozo = linea.strip()
        i = trozo.lower().find(h.lower())
        usos[h.lower()].append(("CSS/HTML", f"l.{n}: …{trozo[max(0, i - 55):i + 12]}…"))

print("=" * 100, "\nDÓNDE SE USA CADA COLOR\n" + "=" * 100)
for h in sorted(usos, key=lambda h: (list(ROLES).index(ROL_DE[h]) if h in ROL_DE else 99, h)):
    rol = ROL_DE.get(h, "— sin rol en la tabla —")
    print(f"\n{h}  [{rol}{' · ' + NOMBRE[h] if h in NOMBRE else ''}]  ({len(usos[h])} usos)")
    vistos = set()
    for donde, que in usos[h]:
        if (donde, que) in vistos:
            continue
        vistos.add((donde, que))
        print(f"   {donde:28s} {que}")

# 3 · chequeos de la consigna
print("\n" + "=" * 100 + "\nCHEQUEOS\n" + "=" * 100)
problemas = []
for h, lista in usos.items():
    for donde, que in lista:
        if "escala: «" in que:
            cat = que.split("«")[1].rstrip("»")
            esperado = {"Mil-Spec Grade": "#4b69ff"}.get(cat)
            if h == "#4b69ff" and cat != "Mil-Spec Grade":
                problemas.append(f"#4b69ff pinta «{cat}» en {donde}")
            if h == "#e8962e" and cat not in ("elegido",):
                problemas.append(f"#e8962e pinta «{cat}» en {donde}")
for donde, que in usos.get("#4b69ff", []):
    if donde.startswith("ESPECS") and "escala" not in que:
        problemas.append(f"#4b69ff fuera de la escala de rareza: {donde} {que}")
for donde, que in usos.get("#e8962e", []):
    if donde.startswith("ESPECS") and "escala" not in que and donde.split(".")[1] not in ("modelo", "comparacion", "importancia", "proba"):
        problemas.append(f"#e8962e en un gráfico que no es del modelo: {donde} {que}")
print("#4b69ff sólo como Mil-Spec y #e8962e sólo como modelo/predicción:", "OK" if not problemas else "FALLA")
for p in problemas:
    print("   ", p)
viejos = [h for h in ("#ff7a59", "#2fae9f", "#7e8ce0", "#9a8f86", "#c99a4b", "#7d8ea6", "#4f86c6", "#b51fc4", "#d13b3b", "#3550e0")
          if h in usos]
print("colores viejos que ya no deberían aparecer:", viejos or "ninguno")
print("--blue en el CSS:", "sigue" if "--blue" in resto else "no está")


# 4 · CIEDE2000 entre colores de roles distintos
def lab(hexa):
    rgb = [int(hexa[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    x = (0.4124564 * lin[0] + 0.3575761 * lin[1] + 0.1804375 * lin[2]) / 0.95047
    y = (0.2126729 * lin[0] + 0.7151522 * lin[1] + 0.0721750 * lin[2])
    z = (0.0193339 * lin[0] + 0.1191920 * lin[1] + 0.9503041 * lin[2]) / 1.08883
    f = lambda t: t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116
    return 116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z))


def ciede2000(l1, l2):
    L1, a1, b1 = l1; L2, a2, b2 = l2
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if C1p else 0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if C2p else 0
    dLp, dCp = L2 - L1, C2p - C1p
    dh = 0 if C1p * C2p == 0 else (h2p - h1p if abs(h2p - h1p) <= 180 else h2p - h1p - 360 if h2p - h1p > 180 else h2p - h1p + 360)
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dh / 2))
    Lpm, Cpm = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hpm = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hpm = (h1p + h2p) / 2
    else:
        hpm = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    T = (1 - 0.17 * math.cos(math.radians(hpm - 30)) + 0.24 * math.cos(math.radians(2 * hpm))
         + 0.32 * math.cos(math.radians(3 * hpm + 6)) - 0.20 * math.cos(math.radians(4 * hpm - 63)))
    dtheta = 30 * math.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * math.sqrt(Cpm ** 7 / (Cpm ** 7 + 25 ** 7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / math.sqrt(20 + (Lpm - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cpm, 1 + 0.015 * Cpm * T
    Rt = -math.sin(math.radians(2 * dtheta)) * Rc
    return math.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


# comprobación de la fórmula con un par de los datos de prueba de Sharma, Wu y Dalal (2005)
assert abs(ciede2000((50, 2.6772, -79.7751), (50, 0, -82.7485)) - 2.0425) < 1e-3
assert abs(ciede2000((50, -1.3802, -84.2814), (50, 0, -82.7485)) - 1.0000) < 1e-3

print("\nCIEDE2000 entre colores de ROLES DISTINTOS (umbral 15; gris contra Consumer exceptuado):")
pares = []
for (r1, c1s), (r2, c2s) in itertools.combinations(ROLES.items(), 2):
    if {r1, r2} == {"modelo"}:
        continue
    for c1 in c1s:
        for c2 in c2s:
            pares.append((ciede2000(lab(c1), lab(c2)), c1, c2))
falla = []
for d, c1, c2 in sorted(pares):
    exento = {c1, c2} == {"#919191", "#9fb3cc"}
    marca = "exento" if exento else ("OK" if d >= 15 else "FALLA")
    if marca == "FALLA":
        falla.append((d, c1, c2))
    if d < 25 or marca != "OK":
        print(f"   {d:5.1f}  {c1} {NOMBRE.get(c1, ''):17s} vs {c2} {NOMBRE.get(c2, ''):17s} {marca}")
print(f"   ({len(pares)} pares; se muestran los menores a 25)")
print("Resultado deltaE:", "OK, todos ≥ 15" if not falla else f"FALLA en {len(falla)} pares")
