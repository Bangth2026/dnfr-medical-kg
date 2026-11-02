"""Parse the health knowledge graph of Rotmensch et al. (2017) (noisy-OR model) from the
layout text of its Supplementary Information into a list [disease, symptom, importance].

Usage:  pdftotext -layout <supplementary.pdf> hkg.txt
        python build_hkg.py hkg.txt hkg_edges.json
Source: Rotmensch M, Halpern Y, Tlimat A, Horng S, Sontag D. Learning a health knowledge
graph from electronic medical records. Sci Rep 2017;7:5994. doi:10.1038/s41598-017-05778-z
(open access, CC BY 4.0).
"""
import re, sys, json

ITEM = re.compile(r"([^,()]+?)\s*\((0\.\d+)\)")


def split(ln):
    if ln.startswith(" "):
        return "", ln.strip()
    m = re.match(r"(\S.*?)\s{3,}(.*)$", ln)
    return (m.group(1).strip(), m.group(2).strip()) if m else (ln.strip(), "")


def parse(lines):
    rows, right, name, on = [], [], [], False
    for ln in lines:
        ln = ln.rstrip("\n").replace("\f", "")
        if not ln.strip():
            continue
        left, txt = split(ln)
        if not on:
            if "(0." in txt:
                on = True
            else:
                continue
        if left and not txt and "(0." not in ln and name == [] and right == []:
            break                              # first line after the table
        if not left and name:                  # a new block starts: close the previous one
            flush(rows, " ".join(name), right); right, name = [], []
        if left and name and not txt and re.match(r"(Appendix|Figure|Table)\b", left):
            break                              # text after the table
        if left:
            name.append(left)
        if txt:
            right.append(txt)
    if name:
        flush(rows, " ".join(name), right)
    return rows


def flush(rows, disease, right):
    text = re.sub(r"\(\s*(0\.[\d\s]*?)\s*\)", lambda m: "(" + m.group(1).replace(" ", "") + ")", " ".join(right))
    for s, w in ITEM.findall(text):
        rows.append([disease, re.sub(r"\s+", " ", s.strip(" ,")), float(w)])


if __name__ == "__main__":
    rows = parse(open(sys.argv[1], encoding="utf-8").readlines())
    json.dump(rows, open(sys.argv[2], "w"))
    print(len(rows), "rows,", len({r[0] for r in rows}), "diseases,", len({r[1] for r in rows}), "symptoms")
