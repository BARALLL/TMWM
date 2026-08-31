"""
Shared configuration: sampling parameters, stage-cache helpers, the
canonical RunConfig used to make checkpoints self-describing, and
CLI argument validation.

Single source of truth for:
  - DT (raw CSV timestep)
  - HDF5_RAW_KEYS (column names in segment HDF5 groups)
  - SamplingConfig (window_len + subsample_factor bundle)
  - Pipeline stage cache helpers (stage_is_stale / save_stage_meta)

Every other module imports these from here — never re-defines them.
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

DT = 0.01  # raw CSV timestep (100 Hz)

# Column names inside HDF5 segment groups (must match ingest_raw.py)
HDF5_RAW_KEYS = (
    'pos', 'quat', 'vel_world', 'angvel_world',
    'damper_len', 'wheel_rot_spd', 'cur_gear',
    'input_steer', 'input_gas', 'input_brake',
)

import random

def set_seed(seed: int) -> None:
    """
    Seed every RNG this pipeline touches at process start, for
    reproducibility across identical CLI invocations.

    Covers: Python's `random` (used nowhere critical today, harmless to
    seed anyway), numpy's global RNG, torch's CPU RNG (model weight
    init — nn.Linear/GRUCell/Embedding all draw from it at
    construction — and DataLoader shuffling order, since RandomSampler
    uses the global default generator unless one is passed explicitly),
    and torch's CUDA RNG on every visible device.

    Deliberately does NOT touch the independent
    `np.random.default_rng(seed)` instances used by split_replays and
    preprocess_geometry's point sampling — those already carry their
    own explicit seed parameter and are unaffected by (and don't
    interact with) the global numpy RNG state seeded here.

    Does NOT by itself make DataLoader worker order reproducible when
    num_workers > 0 — worker processes get their own RNG state on
    fork/spawn; see dataset.py's worker_init_fn for that half.

    Does NOT guarantee bit-exact reproducibility on GPU (cuDNN/cuBLAS
    algorithm selection can still introduce nondeterminism for some
    ops). That requires torch.use_deterministic_algorithms(True) plus
    environment/config changes with a real performance cost, and is
    out of scope for the "identical CLI args → same run" guarantee
    this function provides.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────
# Sampling configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SamplingConfig:
    """Bundles the two parameters that jointly determine which frames
    the model sees: window_len and subsample_factor.  Passing this
    single object instead of two independent primitives eliminates the
    class of bugs where one value is updated and the other is not."""
    window_len: int
    subsample_factor: int = 1

    @property
    def effective_dt(self) -> float:
        """Seconds between consecutive (possibly subsampled) frames."""
        return get_effective_dt(self.subsample_factor)


    def to_dict(self) -> dict:
        return {'window_len': self.window_len, 'subsample_factor': self.subsample_factor}

    @classmethod
    def from_dict(cls, d: dict) -> 'SamplingConfig':
        return cls(window_len=d['window_len'], subsample_factor=d['subsample_factor'])

def get_effective_dt(subsample_factor: int) -> float:
    """Seconds between consecutive frames after subsampling."""
    return DT * subsample_factor

# ──────────────────────────────────────────────────────────────────────
# Pipeline stage caching
# ──────────────────────────────────────────────────────────────────────

def load_stage_meta(meta_path: Path) -> dict | None:
    """Load a stage's sidecar metadata, or None if it doesn't exist."""
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


def stage_is_stale(meta_path: Path, current_key: dict) -> bool:
    """True if no cached meta exists, or any identity field differs.

    Ignores extra (diagnostic) fields in the cached meta — robust to
    the writer adding new informational fields later without this
    check needing to change."""
    prev = load_stage_meta(meta_path)
    if prev is None:
        return True
    return {k: prev.get(k) for k in current_key} != current_key


def save_stage_meta(meta_path: Path, key: dict, **diagnostics) -> None:
    """Write stage identity + optional diagnostic fields to a sidecar JSON."""
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump({**key, **diagnostics}, f, indent=2)

# ─────────────────────────────────────────────────────────────────────
# Run configuration
# ─────────────────────────────────────────────────────────────────────

# DATA_FIELDS: changing these means "same weights, different meaning of
# the inputs" — silently wrong, not a crash. Overridable via
# --resume-force if you're sure it's fine.
#
# ARCH_FIELDS: changing these means the state_dict shapes won't match —
# always fatal, --resume-force cannot help. num_materials specifically
# isn't knowable until the geometry/dataset stages have run, so it's
# split out into EARLY_ARCH_FIELDS (checkable immediately) vs the full
# ARCH_FIELDS (checked once num_materials is known).
DATA_FIELDS = [
    'csv_dir', 'mesh', 'fastest_replay', 'min_segment_length',
    'sample_interval', 'radius', 'n_points', 'val_fraction', 'seed',
]

# ── new: fields recording which run of preprocess_geometry.py's
# material registry, and which features.py schema, produced this
# checkpoint. Both are "changing this silently reinterprets the
# model's inputs/outputs" — same severity class as num_materials.
ARCH_FIELDS = [
    'sampling', 'hidden', 'track_out',
    'num_materials', 'material_registry_hash', 'feature_schema_hash',
]
# num_materials and material_registry_hash both require the geometry
# stage to have run before they're knowable; feature_schema_hash is
# knowable immediately (pure function of features.py's current state).
LATE_ARCH_FIELDS = ['num_materials', 'material_registry_hash']
EARLY_ARCH_FIELDS = [f for f in ARCH_FIELDS if f not in LATE_ARCH_FIELDS]
IMMUTABLE_FIELDS = DATA_FIELDS + ARCH_FIELDS

# Fields that describe *how training proceeds* and are safe to inherit
# from a checkpoint when not explicitly re-passed (e.g. extending
# --epochs, adjusting --rollout-end for a curriculum, dropping --lr
# mid-run). Purely about training *schedule*, not the machine it runs on.
SCHEDULE_FIELDS = [
    'epochs', 'lr', 'weight_decay', 'rollout_start', 'rollout_end',
    'tf_start', 'checkpoint_every',
    'lr_warmup_epochs', 'lr_min_ratio',
]

# Fields that describe *where and how fast* training runs on THIS
# invocation. These must NEVER be silently inherited from a checkpoint:
#   - device: a checkpoint saved with --device cuda could point at a
#     GPU that doesn't exist on the machine resuming it (or the wrong
#     one, in a multi-GPU box) — this should fail loudly or just use
#     what the current invocation resolved, never a stale baked-in value.
#   - batch_size / num_workers: sized for the machine/GPU memory at
#     save time; blindly inheriting them onto a different machine can
#     OOM the GPU or thrash CPU with too many workers.
# These always take the CURRENT invocation's value — explicit flag or
# today's argparse default — exactly like output_dir: never subject to
# checkpoint inheritance or mismatch-checking.
RUNTIME_FIELDS = ['batch_size', 'num_workers', 'device']

# Kept for callers that want "every mutable (non-identity) field" as a
# single set (e.g. documentation/introspection). The resume-merge logic
# itself (merge_mutable_fields_into_args) iterates SCHEDULE_FIELDS and
# RUNTIME_FIELDS separately, since they're handled very differently.
MUTABLE_FIELDS = SCHEDULE_FIELDS + RUNTIME_FIELDS


@dataclass
class RunConfig:
    """
    Canonical, serializable description of everything needed to
    reproduce or resume a training run. Embedded whole in every
    checkpoint so a checkpoint file is self-describing.
    """
    # data-defining
    csv_dir: str
    mesh: str
    fastest_replay: str
    min_segment_length: int
    sample_interval: float
    radius: float
    n_points: int
    val_fraction: float
    seed: int

    # architecture-defining
    sampling: SamplingConfig
    hidden: int
    track_out: int
    num_materials: int
    material_registry_hash: str
    feature_schema_hash: str

    # runtime / schedule (mutable across resume)
    epochs: int
    batch_size: int
    weight_decay: float
    lr: float
    lr_warmup_epochs: int
    lr_min_ratio: float
    rollout_start: int
    rollout_end: int
    tf_start: float
    checkpoint_every: int
    num_workers: int
    device: str

    @classmethod
    def from_args(
        cls,
        args,
        num_materials: int,
        material_registry_hash: str,
        feature_schema_hash: str,
    ) -> "RunConfig":
        return cls(
            csv_dir=str(Path(args.csv_dir).resolve()),
            mesh=str(Path(args.mesh).resolve()),
            fastest_replay=str(Path(args.fastest_replay).resolve()),
            min_segment_length=args.min_segment_length,
            sample_interval=args.sample_interval,
            radius=args.radius,
            n_points=args.n_points,
            val_fraction=args.val_fraction,
            seed=args.seed,
            sampling=SamplingConfig(window_len=args.window_len, subsample_factor=args.subsample_factor),
            hidden=args.hidden,
            track_out=args.track_out,
            num_materials=num_materials,
            material_registry_hash=material_registry_hash,
            feature_schema_hash=feature_schema_hash,
            epochs=args.epochs,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            lr=args.lr,
            lr_warmup_epochs=args.lr_warmup_epochs,
            lr_min_ratio=args.lr_min_ratio,
            rollout_start=args.rollout_start,
            rollout_end=args.rollout_end,
            tf_start=args.tf_start,
            checkpoint_every=args.checkpoint_every,
            num_workers=args.num_workers,
            device=str(args.device),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d['sampling'] = self.sampling.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'RunConfig':
        d = dict(d)
        d['sampling'] = SamplingConfig.from_dict(d['sampling'])
        return cls(**d)

    def diff(self, other: 'RunConfig', fields: list[str]) -> dict[str, tuple[Any, Any]]:
        """
        Compare `self` (e.g. loaded from a checkpoint) against `other`
        (e.g. built from the current CLI args) over `fields`.
        Returns {field: (self_value, other_value)} for every mismatch.
        """
        mismatches = {}
        for f in fields:
            a = getattr(self, f)
            b = getattr(other, f)
            a_cmp = a.to_dict() if isinstance(a, SamplingConfig) else a
            b_cmp = b.to_dict() if isinstance(b, SamplingConfig) else b
            if a_cmp != b_cmp:
                mismatches[f] = (a, b)
        return mismatches


# ─────────────────────────────────────────────────────────────────────
# Explicit-CLI-flag detection (for resume mutable-field merging)
# ─────────────────────────────────────────────────────────────────────

def get_explicit_cli_args(
    parser: argparse.ArgumentParser, argv: list[str] | None = None,
) -> set[str]:
    """
    Return the set of `dest` names actually passed on the command line,
    as distinct from ones filled in by argparse defaults.

    Implemented by re-parsing the same argv against a shadow copy of
    the parser where every default is replaced by a unique sentinel —
    any resulting attribute that isn't the sentinel was explicitly
    passed. This delegates the actual tokenizing to argparse itself,
    so `--flag value`, `--flag=value`, and `store_true` flags are all
    handled correctly and identically to how the real parser sees them
    (unlike matching against raw `sys.argv` strings by hand, which
    can't reliably distinguish `--flag=value` from `--flag` followed by
    an unrelated token).
    """
    sentinel = object()
    shadow = argparse.ArgumentParser(add_help=False)
    for action in parser._actions:
        if not action.option_strings or isinstance(action, argparse._HelpAction):
            continue
        kwargs: dict[str, Any] = {
            'dest': action.dest, 'default': sentinel, 'required': False,
        }
        if isinstance(action, argparse._StoreTrueAction):
            kwargs['action'] = 'store_true'
        elif isinstance(action, argparse._StoreFalseAction):
            kwargs['action'] = 'store_false'
        else:
            kwargs['type'] = action.type if action.type is not None else str
            if action.nargs is not None:
                kwargs['nargs'] = action.nargs
        shadow.add_argument(*action.option_strings, **kwargs)
    ns, _ = shadow.parse_known_args(argv)
    return {name for name, value in vars(ns).items() if value is not sentinel}


def merge_mutable_fields_into_args(
    args: argparse.Namespace, old_config: RunConfig, explicit: set[str],
) -> tuple[dict, dict, dict]:
    """
    Mutates `args` in place.

    SCHEDULE_FIELDS — for each entry:
      - if explicitly passed on the CLI this invocation, leave it alone
        (explicit always wins)
      - otherwise, overwrite it with the checkpoint's original value,
        rather than silently letting it sit at today's argparse default

    RUNTIME_FIELDS — NEVER inherited from the checkpoint, regardless of
    whether they were explicitly passed this invocation. They always
    keep whatever value this invocation resolved to (explicit flag or
    current default) — see RUNTIME_FIELDS' comment for why.

    Must be called *before* anything downstream reads `args` (dataloader
    construction, device resolution, RunConfig.from_args) — otherwise
    an inherited rollout_end/lr/etc. won't actually take effect.

    Returns (overrides_applied, inherited, runtime_diffs):
      - overrides_applied: SCHEDULE_FIELDS explicitly passed this run
      - inherited:         SCHEDULE_FIELDS pulled from the checkpoint
      - runtime_diffs:     RUNTIME_FIELDS where the checkpoint's
        recorded value differs from what this invocation is actually
        using — informational only (never applied to `args`), so a
        resume onto a different machine/config is visible in logs
        instead of silently invisible.
    """
    overrides_applied: dict = {}
    inherited: dict = {}
    runtime_diffs: dict = {}

    for f in SCHEDULE_FIELDS:
        if f in explicit:
            overrides_applied[f] = getattr(args, f)
        else:
            inherited_value = getattr(old_config, f)
            if getattr(args, f) != inherited_value:
                inherited[f] = inherited_value
                setattr(args, f, inherited_value)

    for f in RUNTIME_FIELDS:
        current_value = getattr(args, f)
        checkpoint_value = getattr(old_config, f)
        if current_value != checkpoint_value:
            runtime_diffs[f] = (checkpoint_value, current_value)
        # deliberately no setattr: current_value always wins

    return overrides_applied, inherited, runtime_diffs


# ─────────────────────────────────────────────────────────────────────
# CLI argument validation
# ─────────────────────────────────────────────────────────────────────

def validate_args(args) -> list[str]:
    """
    Sanity-check CLI args before any expensive work happens. Returns a
    list of human-readable problem descriptions (empty if none found).
    Also re-run after merging resume mutable-field overrides, since
    combining two independently-valid sources (checkpoint + explicit
    CLI overrides) can still produce an invalid combination.
    """
    issues = []

    def require(cond: bool, msg: str):
        if not cond:
            issues.append(msg)

    require(args.window_len >= 1, f'--window-len must be >= 1 (got {args.window_len})')
    require(args.subsample_factor >= 1,
            f'--subsample-factor must be >= 1 (got {args.subsample_factor})')

    require(args.rollout_start >= 1,
            f'--rollout-start must be >= 1 (got {args.rollout_start})')
    require(args.rollout_end >= args.rollout_start,
            f'--rollout-end ({args.rollout_end}) must be >= --rollout-start ({args.rollout_start})')
    require(args.rollout_end <= args.window_len,
            f'--rollout-end ({args.rollout_end}) must be <= --window-len ({args.window_len})')

    require(0.0 <= args.tf_start <= 1.0,
            f'--tf-start must be in [0, 1] (got {args.tf_start})')

    require(0.0 < args.val_fraction < 1.0,
            f'--val-fraction must be in (0, 1) (got {args.val_fraction})')

    require(args.epochs >= 1, f'--epochs must be >= 1 (got {args.epochs})')
    require(args.batch_size >= 1, f'--batch-size must be >= 1 (got {args.batch_size})')
    require(args.weight_decay >= 0, f'--weight-decay must be >= 0 (got {args.weight_decay})')
    require(args.checkpoint_every >= 1,
            f'--checkpoint-every must be >= 1 (got {args.checkpoint_every})')
    require(args.num_workers >= 0, f'--num-workers must be >= 0 (got {args.num_workers})')

    require(args.min_segment_length >= 1,
            f'--min-segment-length must be >= 1 (got {args.min_segment_length})')
    require(args.n_points >= 1, f'--n-points must be >= 1 (got {args.n_points})')
    require(args.sample_interval > 0,
            f'--sample-interval must be > 0 (got {args.sample_interval})')
    require(args.radius > 0, f'--radius must be > 0 (got {args.radius})')
    require(args.lr > 0, f'--lr must be > 0 (got {args.lr})')
    require(args.lr_warmup_epochs >= 0,
            f'--lr-warmup-epochs must be >= 0 (got {args.lr_warmup_epochs})')
    require(0.0 < args.lr_min_ratio <= 1.0,
            f'--lr-min-ratio must be in (0, 1] (got {args.lr_min_ratio})')
    require(args.hidden >= 1, f'--hidden must be >= 1 (got {args.hidden})')
    require(args.track_out >= 1, f'--track-out must be >= 1 (got {args.track_out})')

    if args.resume is not None:
        require(Path(args.resume).exists(), f'--resume path does not exist: {args.resume}')

    return issues
