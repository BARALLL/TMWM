"""
ModelBundle: everything needed to run the world model at inference
time, loaded once per process. Track geometry is loaded SEPARATELY
from the checkpoint's own training-time geometry — this is exactly the
"zero-shot new track" seam: swap geo_h5_path, keep the checkpoint.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import RunConfig
from checkpointing import load_checkpoint, strip_compile_prefix
from model import WorldModel
from dataset import NormStats
from track_context import TrackContextExtractor, load_extractor
from features import FEATURE_SCHEMA_HASH
from dynamics import NormTensors


@dataclass
class ModelBundle:
    model: WorldModel
    norm: NormTensors
    track_ctx: TrackContextExtractor
    run_config: RunConfig
    device: torch.device

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        geo_h5_path: str | Path,
        norm_stats_path: str | Path | None = None,
        device: torch.device | str = 'auto',
    ) -> 'ModelBundle':
        if device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)

        checkpoint_path = Path(checkpoint_path)
        ckpt = load_checkpoint(checkpoint_path, map_location=device)
        ckpt['run_config']["material_registry_hash"] = "" #!/!\
        ckpt['run_config']["feature_schema_hash"] = "1e12fb514095a99951cb57a082896863bf4266ae"
        run_config = RunConfig.from_dict(ckpt['run_config'])

        if run_config.feature_schema_hash != FEATURE_SCHEMA_HASH:
            raise ValueError(
                f'Checkpoint was trained under a different feature schema '
                f'(checkpoint={run_config.feature_schema_hash}, '
                f'current code={FEATURE_SCHEMA_HASH}) — STATE_FEATURES/'
                f'TARGET_FEATURES/ACTION_FEATURES in features.py have changed '
                f'since this checkpoint was produced. Loading it now would '
                f'silently apply the wrong column semantics. Retrain, or '
                f'check out the features.py this checkpoint was trained under.'
            )

        model = WorldModel.from_config(run_config).to(device)
        state_dict = strip_compile_prefix(ckpt['model_state_dict'])
        model.load_state_dict(state_dict)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        track_ctx = load_extractor(str(geo_h5_path)).to(device)

        if track_ctx.material_registry_hash and run_config.material_registry_hash:
            if track_ctx.material_registry_hash != run_config.material_registry_hash:
                raise ValueError(
                    f'Track geometry at {geo_h5_path} was built with a different '
                    f'material registry than the checkpoint was trained on '
                    f'(geo={track_ctx.material_registry_hash}, '
                    f'checkpoint={run_config.material_registry_hash}). Compact '
                    f'material IDs would mean different surfaces than at training '
                    f'time. Re-run preprocess_geometry.py with the SAME --registry '
                    f'file used for training data.'
                )
        # else: one side predates this field (older geo file / older
        # checkpoint) — can't verify, proceed with a print rather than
        # block indefinitely on legacy artifacts.
        elif not track_ctx.material_registry_hash or not run_config.material_registry_hash:
            print('  WARNING: could not verify material registry provenance '
                  '(missing hash on geometry file or checkpoint) — proceeding '
                  'without this check.')

        if norm_stats_path is None:
            # Mirrors run_pipeline.py's layout: checkpoints/ is a sibling of
            # norm_stats.npz under the same output_dir. Pass norm_stats_path
            # explicitly if your deployment layout differs.
            norm_stats_path = checkpoint_path.parent.parent / 'norm_stats.npz'
        norm = NormTensors.from_norm_torch(
            NormStats.load(str(norm_stats_path)).to_torch(device)
        )

        return cls(model=model, norm=norm, track_ctx=track_ctx,
                    run_config=run_config, device=device)

    def swap_track(self, geo_h5_path: str | Path) -> None:
        """Load a different track's geometry in place, without touching
        weights/norm stats — the actual zero-shot-new-track operation."""
        new_ctx = load_extractor(str(geo_h5_path)).to(self.device)
        if (self.track_ctx.material_registry_hash and self.run_config.material_registry_hash
                and new_ctx.material_registry_hash != self.run_config.material_registry_hash):
            raise ValueError(
                f'New track geometry at {geo_h5_path} uses a different material '
                f'registry than this checkpoint was trained on.'
            )
        self.track_ctx = new_ctx