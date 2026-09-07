"""临时: 全模块导入自检脚本(用后会删除)"""
import importlib
import pathlib
import sys

root = pathlib.Path(r"D:\pythoncharm\pythonProject\atguigu_edu_keeper")
skip = {"main", "api.query_router"}
mods = set()
for p in root.rglob("*.py"):
    parts = list(p.relative_to(root).with_suffix("").parts)
    if parts[0] in (".venv", ".idea", "docs") or "__pycache__" in parts or "test" in parts:
        continue
    mods.add(".".join(parts))

mods = sorted(mods - skip)
ok, fail = 0, []
for m in mods:
    try:
        importlib.import_module(m)
        ok += 1
    except Exception as e:  # noqa: BLE001
        fail.append((m, type(e).__name__, str(e)[:250]))

print(f"total={len(mods)} ok={ok} fail={len(fail)}")
for m, t, msg in fail:
    print(f"FAIL {m}: [{t}] {msg}")
