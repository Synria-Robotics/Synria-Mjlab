# Synria Mjlab

Synria robot learning tasks built with [mjlab](https://github.com/mujocolab/mjlab).
Robot MJCF/URDF assets are loaded from [synriard](https://github.com/Synria-Robotics/Synria-robot-descriptions) (no vendored XML).

Task layout follows [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) taxonomy
(`manipulation/`, `locomotion/`), with mjlab manager-based registration.

## Tasks

| Task ID | Robot | Description |
|---------|-------|-------------|
| `Reach-Alicia-D` | Alicia_D | Move EE to target pose |
| `Reach-Alicia-M` | Alicia_M | Move EE to target pose |
| `Lift-Alicia-D` | Alicia_D | Lift a cube |
| `Lift-Alicia-M` | Alicia_M | Lift a cube |
| `HandOver-Bessica-D` | Bessica_D | Dual-arm object handover |
| `HandOver-Bessica-M` | Bessica_M | Dual-arm object handover |
| `PegInsertion-Bessica-D` | Bessica_D | Dual-arm peg insertion |
| `PegInsertion-Bessica-M` | Bessica_M | Dual-arm peg insertion |
| `Velocity-Flat-Corina` | Corina | Flat-terrain velocity tracking |
| `Getup-Flat-Corina` | Corina | Fall recovery on flat terrain |

## Setup

```bash
cd Synria-Mjlab
uv sync --extra cu128
```

## Train / play

```bash
uv run train Reach-Alicia-D --num_envs 4096
uv run play Reach-Alicia-D
uv run train Lift-Alicia-D --num_envs 4096
```

Assets resolve via:

```python
from synriard import get_model_path
path = get_model_path("Alicia_D", version="v5_6", variant="gripper_50mm", model_format="mjcf")
```
