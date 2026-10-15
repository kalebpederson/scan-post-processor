"""Poor-man's line profiler for clean.clean_page: .venv/bin/python bench/lprof.py [name] [dpi]"""
import cv2, time, sys, collections
cv2.setNumThreads(1)
from scanpp import clean
name = sys.argv[1] if len(sys.argv) > 1 else "pair02_front"
dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 300
im = cv2.cvtColor(cv2.imread(f"research/corpus/{name}.jpg"), cv2.COLOR_BGR2RGB)
if dpi != 300: im = cv2.resize(im, None, fx=dpi / 300, fy=dpi / 300, interpolation=cv2.INTER_AREA)
T = collections.Counter(); last = [None, 0.0]
code = clean.clean_page.__code__
def tr(frame, ev, arg):
    if frame.f_code is not code: return None
    def lt(fr, ev, arg):
        now = time.process_time()
        if last[0] is not None: T[last[0]] += now - last[1]
        last[0] = fr.f_lineno if ev == "line" else None; last[1] = time.process_time()
        return lt
    return lt
clean.clean_page(im, dpi)
N = 3
sys.settrace(tr)
for _ in range(N): clean.clean_page(im, dpi)
sys.settrace(None)
src = open(clean.__file__).read().split("\n")
tot = sum(T.values())
for ln, t in sorted(T.items(), key=lambda kv: -kv[1])[:22]:
    print(f"{t / N * 1000:7.1f} ms  L{ln}: {src[ln - 1].strip()[:90]}")
print("total", tot / N * 1000)
