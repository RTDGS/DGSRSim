import sys
import importlib.util

def where(pkg_name: str):
    spec = importlib.util.find_spec(pkg_name)
    print(f"{pkg_name}.spec =", spec)
    if spec is None:
        return
    print(f"{pkg_name}.origin =", spec.origin)
    print(f"{pkg_name}.submodule_search_locations =", list(spec.submodule_search_locations or []))

print("python:", sys.executable)
print("cwd:", __import__("os").getcwd())
print("----")
where("leisaac")
where("leisaac.devices")
