import yaml
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "configs/baseline.yaml"

with open(path) as f:
    cfg = yaml.safe_load(f)

print("Parsed successfully:\n")
print(cfg)
