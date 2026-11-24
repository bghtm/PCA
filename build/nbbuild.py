"""把 (类型, 源码) 序列写成 nbformat v4 notebook。"""
import nbformat as nbf


def build(cells, path, kernel="python3"):
    nb = nbf.v4.new_notebook()
    nb.cells = []
    for kind, src in cells:
        s = src.strip("\n")
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(s))
        elif kind == "code":
            nb.cells.append(nbf.v4.new_code_cell(s))
        else:
            raise ValueError(kind)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.13"}
    nbf.write(nb, path)
    print("wrote", path, len(nb.cells), "cells")
