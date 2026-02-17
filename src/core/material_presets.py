"""Material presets for common brittle/ductile materials.

Each preset maps a material name to physical parameters used by the
elasticity model and AT2 phase field solver.

Usage in config:
    material:
      preset: "concrete"    # applies all preset values
      youngs_modulus: 1e8   # optional: override individual params
"""

MATERIAL_PRESETS = {
    "concrete": {
        "youngs_modulus": 30e9,
        "poissons_ratio": 0.2,
        "Gc": 100.0,
        "l0": 0.01,
        "density": 2400.0,
        "description": "Standard concrete (C30/37)",
    },
    "glass": {
        "youngs_modulus": 70e9,
        "poissons_ratio": 0.22,
        "Gc": 8.0,
        "l0": 0.005,
        "density": 2500.0,
        "description": "Soda-lime glass (brittle, low Gc)",
    },
    "ceramic": {
        "youngs_modulus": 300e9,
        "poissons_ratio": 0.25,
        "Gc": 30.0,
        "l0": 0.008,
        "density": 3900.0,
        "description": "Alumina ceramic (Al2O3)",
    },
    "wood": {
        "youngs_modulus": 12e9,
        "poissons_ratio": 0.3,
        "Gc": 300.0,
        "l0": 0.02,
        "density": 600.0,
        "description": "Softwood (pine, along grain)",
    },
    "steel": {
        "youngs_modulus": 200e9,
        "poissons_ratio": 0.3,
        "Gc": 50000.0,
        "l0": 0.005,
        "density": 7800.0,
        "description": "Structural steel (ductile, very high Gc)",
    },
}

# Keys that get applied from preset to config.material
_PRESET_KEYS = ["youngs_modulus", "poissons_ratio", "Gc", "l0", "density"]


def resolve_material_preset(config):
    """Apply material preset values to config, allowing individual overrides.

    If config.material.preset is set to a valid preset name, the preset
    values are written into config.material. Any values explicitly specified
    alongside the preset in the YAML will override the preset defaults.

    Args:
        config: OmegaConf configuration object (modified in-place)

    Returns:
        config (same object, for chaining)
    """
    from omegaconf import OmegaConf

    preset_name = config.material.get("preset", None)
    if preset_name is None:
        return config

    if preset_name not in MATERIAL_PRESETS:
        available = ", ".join(sorted(MATERIAL_PRESETS.keys()))
        raise ValueError(
            f"Unknown material preset: '{preset_name}'. "
            f"Available: {available}"
        )

    preset = MATERIAL_PRESETS[preset_name]
    print(f"[Material] Applying preset: '{preset_name}' — {preset['description']}")

    for key in _PRESET_KEYS:
        OmegaConf.update(config, f"material.{key}", preset[key])

    print(
        f"  E={config.material.youngs_modulus:.2e}, "
        f"nu={config.material.poissons_ratio}, "
        f"Gc={config.material.Gc}, "
        f"l0={config.material.l0}, "
        f"rho={config.material.density}"
    )

    return config


def validate_l0(config):
    """Warn and auto-adjust if l0 < 2*dx (unresolvable damage band)."""
    num_grids = config.mpm.get("num_grids", 64)
    dx = 1.0 / num_grids
    min_l0 = 2.0 * dx
    l0 = config.material.get("l0", 0.035)

    if l0 < min_l0:
        print(
            f"[Warning] l0={l0:.4f} < 2*dx={min_l0:.4f} "
            f"(grid={num_grids}). Adjusting l0 to {min_l0:.4f}"
        )
        from omegaconf import OmegaConf
        OmegaConf.update(config, "material.l0", min_l0)

    return config
