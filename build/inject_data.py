# -*- coding: utf-8 -*-
"""把鸢尾花数据注入 03_iris_pca_可视化.html 的 IRIS-DATA 区块，并输出期望值供 Node 对拍。"""
import io
import json
import os
import re

from sklearn.datasets import load_iris

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = os.path.join(ROOT, "03_iris_pca_可视化.html")

iris = load_iris()
X = iris.data
y = iris.target
features = ["花萼长 SL (cm)", "花萼宽 SW (cm)", "花瓣长 PL (cm)", "花瓣宽 PW (cm)"]
species = [str(s) for s in iris.target_names]

rows = ",\n".join("    [" + ", ".join(f"{v:.1f}" for v in row) + "]" for row in X)
ys = ", ".join(str(int(t)) for t in y)

block = (
    "/* IRIS-DATA-BEGIN */\n"
    "const IRIS = {\n"
    "  features: " + json.dumps(features, ensure_ascii=False) + ",\n"
    "  species: " + json.dumps(species, ensure_ascii=False) + ",\n"
    "  x: [\n" + rows + "\n  ],\n"
    "  y: [" + ys + "]\n"
    "};\n"
    "/* IRIS-DATA-END */"
)

html = io.open(HTML, encoding="utf-8").read()
pattern = re.compile(r"/\* IRIS-DATA-BEGIN \*/.*?/\* IRIS-DATA-END \*/", re.S)
assert pattern.search(html), "找不到 IRIS-DATA 区块"
html_new = pattern.sub(lambda m: block, html, count=1)
io.open(HTML, "w", encoding="utf-8", newline="\n").write(html_new)
print("注入完成：%d 行 × %d 列" % (X.shape[0], X.shape[1]))
print("文件大小：%.1f KB" % (len(html_new.encode('utf-8')) / 1024))

# ---- 用 numpy 计算期望值，供 node 对拍 ----
import numpy as np


def fit(mat, standardize):
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0, ddof=1) if standardize else np.ones(mat.shape[1])
    Z = (mat - mu) / (sd if standardize else 1.0)
    C = np.cov(Z, rowvar=0)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(-vals)
    vals = vals[order]
    comps = vecs[:, order].T          # 每个主成分一行
    # 符号规范：绝对值最大的分量为正
    for c in range(comps.shape[0]):
        k = int(np.argmax(np.abs(comps[c])))
        if comps[c][k] < 0:
            comps[c] = -comps[c]
    scores = Z @ comps.T
    return {
        "eigvals": vals.tolist(),
        "ratio": (vals / vals.sum()).tolist(),
        "comps": comps.tolist(),
        "scores": scores.tolist(),
        "mu": mu.tolist(),
        "sd": sd.tolist(),
    }


expected = {
    "raw": fit(X, False),
    "std": fit(X, True),
}
out = os.path.join(ROOT, "build", "expected.json")
with io.open(out, "w", encoding="utf-8") as f:
    json.dump(expected, f)
print("期望值写入:", out)
print("raw 特征值:", [round(v, 6) for v in expected["raw"]["eigvals"]])
print("std 特征值:", [round(v, 6) for v in expected["std"]["eigvals"]])