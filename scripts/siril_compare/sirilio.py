"""Minimal FITS I/O + Siril sequence helpers (no astropy)."""
import os, re, subprocess
import numpy as np

SIRIL = ["flatpak", "run", "--filesystem=/tmp/rawsr", "--filesystem=output/siril",
         "--filesystem=data:ro",
         "--command=siril-cli", "org.siril.Siril"]


def _card(k, v):
    if isinstance(v, str):
        s = f"{k:<8}= '{v:<8}'"
    elif isinstance(v, bool):
        s = f"{k:<8}= {'T' if v else 'F':>20}"
    else:
        s = f"{k:<8}= {v:>20}"
    return s.ljust(80)[:80]


def write_cfa_fits(path, mosaic01, bayerpat="BGGR"):
    h, w = mosaic01.shape
    cards = [_card("SIMPLE", True), _card("BITPIX", 16), _card("NAXIS", 2),
             _card("NAXIS1", w), _card("NAXIS2", h), _card("BZERO", 32768),
             _card("BSCALE", 1), _card("BAYERPAT", bayerpat),
             _card("XBAYROFF", 0), _card("YBAYROFF", 0),
             _card("ROWORDER", "TOP-DOWN"), "END".ljust(80)]
    hdr = "".join(cards)
    hdr += " " * ((-len(hdr)) % 2880)
    u = np.clip(np.round(mosaic01 * 65535.0), 0, 65535).astype(np.int32) - 32768
    data = u.astype(">i2").tobytes()
    data += b"\0" * ((-len(data)) % 2880)
    with open(path, "wb") as f:
        f.write(hdr.encode("ascii")); f.write(data)


def read_fits(path):
    raw = open(path, "rb").read()
    hdr = {}
    pos = 0
    while True:
        block = raw[pos:pos + 2880].decode("ascii", "replace"); pos += 2880
        done = False
        for i in range(0, 2880, 80):
            c = block[i:i + 80]
            k = c[:8].strip()
            if k == "END":
                done = True; break
            if c[8:10] == "= ":
                v = c[10:].split("/")[0].strip()
                hdr[k] = v.strip("'").strip()
        if done:
            break
    bp = int(hdr["BITPIX"]); n = int(hdr["NAXIS"])
    dims = [int(hdr[f"NAXIS{i}"]) for i in range(1, n + 1)][::-1]
    dt = {16: ">i2", -32: ">f4", 32: ">i4", 8: "u1", -64: ">f8"}[bp]
    cnt = int(np.prod(dims))
    a = np.frombuffer(raw, dtype=dt, count=cnt, offset=pos).astype(np.float64).reshape(dims)
    a = a * float(hdr.get("BSCALE", 1)) + float(hdr.get("BZERO", 0))
    return a, hdr


def run_siril(workdir, lines, timeout=1800):
    ssf = os.path.join(workdir, "run.ssf")
    with open(ssf, "w") as f:
        f.write("requires 1.4.0\n" + "\n".join(lines) + "\n")
    p = subprocess.run(SIRIL + ["-d", workdir, "-s", ssf], capture_output=True,
                       text=True, timeout=timeout)
    out = p.stdout + p.stderr
    if "Script execution finished successfully" not in out:
        tail = "\n".join(l for l in out.splitlines()[-40:])
        raise RuntimeError("siril failed:\n" + tail)
    return out


def inject_registration(seqpath, homographies, ref=0):
    """Rewrite a Siril .seq with given 3x3 homographies (one per image)."""
    txt = open(seqpath).read().splitlines()
    out = []
    n = len(homographies)
    for l in txt:
        if l.startswith("S "):
            m = re.match(r"S '([^']*)' (.*)", l)
            f = m.group(2).split()
            f[1] = str(n); f[2] = str(n); f[4] = str(ref)
            out.append(f"S '{m.group(1)}' " + " ".join(f))
        elif l.startswith("I ") or l.startswith("R") or l.startswith("L "):
            continue
        else:
            out.append(l)
    # insert after S line
    si = [i for i, l in enumerate(out) if l.startswith("S ")][0]
    ins = ["L 1"] + [f"I {k + 1} 1" for k in range(n)]
    for Hm in homographies:
        hv = " ".join(f"{v:.10g}" for v in np.asarray(Hm, float).ravel())
        ins.append(f"R0 2 2 1 0 1000 100 H {hv}")
    out = out[:si + 1] + ins + [l for l in out[si + 1:]]
    open(seqpath, "w").write("\n".join(out) + "\n")
