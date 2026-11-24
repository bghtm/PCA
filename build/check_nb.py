# -*- coding: utf-8 -*-
"""检查已执行的 notebook：报错、空输出、以及打印关键文本输出。"""
import io
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
path = sys.argv[1]
maxlen = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
nb = json.load(io.open(path, encoding="utf-8"))
n_err = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    outs = c.get("outputs", [])
    kinds = [o.get("output_type") for o in outs]
    print(f"--- cell {i} | outputs={kinds} | exec={c.get('execution_count')}")
    for o in outs:
        if o.get("output_type") == "error":
            n_err += 1
            print("  !! ERROR:", o.get("ename"), o.get("evalue"))
            for line in o.get("traceback", [])[-6:]:
                print("    ", line.replace("\x1b", ""))
        elif o.get("output_type") == "stream":
            txt = "".join(o.get("text", []))
            print("  [stream]")
            print("\n".join("    " + l for l in txt.splitlines()[:40])[:maxlen])
        elif o.get("output_type") == "execute_result":
            txt = "".join(o.get("data", {}).get("text/plain", []))
            print("  [result]", txt[:400].replace("\n", " "))
        elif o.get("output_type") == "display_data":
            keys = list(o.get("data", {}).keys())
            print("  [display]", keys)
print("\nTOTAL ERRORS:", n_err)
