"""Partition group of a study, from its human-coded MeSH field (configs/data/groups_v1.yaml)."""
import yaml


def load_groups(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg["order"][-1] != "other_abnormal" or set(cfg["order"]) != set(cfg["groups"]):
        raise ValueError("group config: 'order' must list every group and end with other_abnormal")
    return cfg


def parse_mesh(mesh):
    """'Cardiomegaly/mild;Aorta/tortuous' -> [('cardiomegaly', ['mild']), ('aorta', ['tortuous'])]."""
    out = []
    for entry in mesh.split(";"):
        parts = [p.strip().lower() for p in entry.split("/")]
        if parts and parts[0]:
            out.append((" ".join(parts[0].split()), [p for p in parts[1:] if p]))
    return out


def assign_group(mesh, cfg):
    entries = parse_mesh(mesh)
    whole = " ".join(mesh.strip().lower().split())
    for name in cfg["order"][:-1]:
        rule = cfg["groups"][name] or {}
        if whole in rule.get("whole_field", []):
            return name
        headings = set(rule.get("headings", []))
        with_attr = rule.get("heading_with_attr", {})
        for heading, attrs in entries:
            if heading in headings:
                return name
            if heading in with_attr and set(with_attr[heading]) & set(attrs):
                return name
    return cfg["order"][-1]
