"""PΛN curriculum token: visual mesh optional, collision is always a plain cylinder."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

from .apex_cfg import COIN_RADIUS, COIN_THICKNESS

# 32 mm x 4 mm PLA cylinder ~ 4 g. density = 1240 kg/m^3.
PAN_COIN_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/object",
    spawn=sim_utils.CylinderCfg(
        radius=COIN_RADIUS,
        height=COIN_THICKNESS,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0025,
            # A coin spawned with any overlap against the knuckles gets launched
            # at 1000 m/s. Cap it so residual overlap resolves gently.
            max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(density=1240.0),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.70,
            dynamic_friction=0.65,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.85, 0.62, 0.18),
            roughness=0.35,
        ),
    ),
    # Hover just above a palm-up hand at z=0.50. Tune after inspect_apex.py.
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.00, 0.045, 0.575), rot=(1.0, 0.0, 0.0, 0.0)),
)
