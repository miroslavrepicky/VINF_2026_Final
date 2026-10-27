import os, glob

def size(paths):
    return sum(os.path.getsize(p) for p in paths)

def mb(b):
    return f"{b / 1024 / 1024:.1f} MB"

html = glob.glob(os.path.join("data", "html", "*.html"))
print("HTML súbory:", len(html), "|", mb(size(html)))

for name in ("drivers.jsonl", "driver_series.jsonl"):
    p = os.path.join("data", name)
    if os.path.exists(p):
        print(name, "|", mb(os.path.getsize(p)))

idx = [os.path.join(r, f) for r, _, fs in os.walk(os.path.join("data", "index")) for f in fs]
print("Index:", len(idx), "súborov |", mb(size(idx)))