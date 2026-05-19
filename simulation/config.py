from pathlib import Path
import tomli as toml

DEFAULTS = {
    "n_simulations": 20,
    "stop_at_n": 100,
    "n_abstracts": [1, 4, 7],
    "length_abstracts": [100, 500, 900],
    "llm_temperature": [0.0, 0.4, 0.8],
    "wss_threshold": 0.95,
    "stimulus_for_llm": ["inclusion_criteria"],
    "subset_datasets": None,
    "condition": None
}

# Resolve pyproject.toml relative to this file's location so the import works
# regardless of the user's working directory.
_DEFAULT_TOML = Path(__file__).parent.parent / "pyproject.toml"

def load_pyproject_config(pyproject_path: Path = _DEFAULT_TOML) -> dict:
    cfg = DEFAULTS.copy()
    if pyproject_path.exists():
        with pyproject_path.open("rb") as f:
            data = toml.load(f)
        cfg.update(data.get("tool", {}).get("jumpstart", {}))
    return cfg
