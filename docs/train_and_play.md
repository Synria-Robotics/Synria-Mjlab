# Train and play

Commands assume you are in the `mjlab` conda env and the `Synria-Mjlab` directory:

```bash
conda activate mjlab
cd /path/to/Synria-Mjlab
```

## List tasks

```bash
list-envs | grep -E 'Reach|Lift|HandOver|PegInsertion|Velocity|Getup'
```

## Train

```bash
train Reach-Alicia-D --env.scene.num-envs 4096
```

Other tasks use the same pattern, e.g. `Lift-Alicia-D`, `HandOver-Bessica-D`.

- Lower `--env.scene.num-envs` (e.g. `1024`) if you run out of GPU memory.
- Logs and checkpoints go to `logs/rsl_rl/<experiment_name>/<timestamp>/`.
- Checkpoints are saved every `save_interval` iterations (default 20), plus a final checkpoint.
- Optional W&B logging uses your existing `wandb` login.

Example log dir:

```text
logs/rsl_rl/reach_alicia_d/2026-07-31_14-41-14/
  model_0.pt
  ...
  model_1499.pt
```

## Play (evaluate a trained policy)

Prefer **Viser** (browser viewer). It works over SSH and exits cleanly with Ctrl+C.

### Local checkpoint

```bash
play Reach-Alicia-D \
  --checkpoint-file logs/rsl_rl/reach_alicia_d/<run>/model_1499.pt \
  --viewer viser \
  --num-envs 1
```

Open the URL printed in the terminal (usually `http://localhost:8080`).

### W&B run

```bash
play Reach-Alicia-D \
  --wandb-run-path <entity>/mjlab/<run_id> \
  --viewer viser \
  --num-envs 1
```

### Record a video

```bash
play Reach-Alicia-D \
  --checkpoint-file logs/rsl_rl/reach_alicia_d/<run>/model_1499.pt \
  --video \
  --video-length 200 \
  --num-envs 1
```

Videos are written under `logs/rsl_rl/.../videos/play/`.

### Useful flags

| Flag | Meaning |
|------|---------|
| `--viewer viser` | Browser viewer (recommended for remote/SSH) |
| `--viewer native` | MuJoCo OpenGL window (local desktop) |
| `--viewer auto` | Native if `DISPLAY`/`WAYLAND_DISPLAY` is set, else Viser |
| `--num-envs N` | Parallel envs to visualize |
| `--checkpoint-file PATH` | Local `.pt` checkpoint |
| `--wandb-run-path ENTITY/PROJECT/RUN_ID` | Load checkpoint from W&B |
| `--video` | Record MP4 during play |

See `play Reach-Alicia-D --help` for the full list.

## Viewer notes (Ctrl+C / GLX)

If `DISPLAY` is set, `--viewer auto` picks the **native** MuJoCo GLX viewer. On remote/X11 setups that often causes:

```text
Ctrl+C received. Shutting down viewer...
X Error of failed request: GLXBadDrawable
```

and the process may hang. Prefer:

```bash
play Reach-Alicia-D --checkpoint-file <ckpt> --viewer viser --num-envs 1
```

If a native viewer is already stuck, force-kill from another terminal:

```bash
pkill -9 -f "play Reach-Alicia-D"
```

## Demo GIF

A short (~5s) Reach demo is in [`media/reach_alicia_d.gif`](../media/reach_alicia_d.gif). Regenerate with:

```bash
python scripts/record_reach_gif.py
```

## Quick reference

```bash
# Train
train Reach-Alicia-D --env.scene.num-envs 4096

# Play (recommended)
play Reach-Alicia-D \
  --checkpoint-file logs/rsl_rl/reach_alicia_d/<run>/model_1499.pt \
  --viewer viser \
  --num-envs 1

# Play from W&B
play Reach-Alicia-D \
  --wandb-run-path <entity>/mjlab/<run_id> \
  --viewer viser \
  --num-envs 1
```
