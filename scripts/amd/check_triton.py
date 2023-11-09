import importlib.util
import sys

module_name = 'triton'

if module_name in sys.modules:
    print(f"'{module_name}' already loaded from {sys.modules[module_name].__file__}")
else:
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        print(f"'{module_name}' found at {spec.origin}")
    else:
        print(f"'{module_name}' not found")
