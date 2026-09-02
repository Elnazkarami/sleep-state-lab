"""The synthetic end-to-end run: every stage, on generated signals, on the CPU.

What it proves and what it does not. It proves the pipeline is connected --
that files are discovered, paired, epoched with their gaps, split without
sharing a participant, normalised from training statistics, trained, checkpointed,
reloaded, predicted, saved and scored. It proves nothing about sleep, and every
artefact it writes is named and labelled synthetic so it cannot be mistaken for
one that is.

It also runs the optimisation check that matters for a new architecture: D1 is
asked to overfit a tiny batch. A network that cannot drive the loss to nearly
zero on sixteen examples it sees over and over has a bug in its gradient path,
and no amount of real data will reveal that as clearly.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

import numpy as np

from sleepstatelab.config import Config, DataConfig, ModelConfig, TrainConfig
from sleepstatelab.labels import STAGES


def overfit_check(
    *, n: int = 16, steps: int = 300, device: str = "cpu", seed: int = 0
) -> dict[str, float]:
    """Can D1 memorise a tiny batch? An optimisation check, not an experiment.

    Five distinguishable synthetic epochs, repeated, trained until the loss
    stops moving. Success is a near-zero loss and perfect accuracy *on the
    training batch itself* -- which is the definition of overfitting and exactly
    what is being asked for.
    """
    import torch
    from torch import nn

    from sleepstatelab.models.d1 import D1Classifier
    from sleepstatelab.training.trainer import seed_everything

    seed_everything(seed)
    rng = np.random.default_rng(seed)
    t = np.arange(3000) / 100.0
    x = np.zeros((n, 2, 3000), dtype=np.float32)
    y = np.array([i % len(STAGES) for i in range(n)], dtype=np.int64)
    for row in range(n):
        frequency = 1.0 + 3.0 * y[row]
        for channel in range(2):
            x[row, channel] = (
                np.sin(2 * np.pi * frequency * t + rng.uniform(0, 6.28))
                + 0.05 * rng.normal(size=3000)
            )

    model = D1Classifier(in_channels=2).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()
    inputs = torch.from_numpy(x).to(device)
    targets = torch.from_numpy(y).to(device)

    model.train()
    first = last = float("nan")
    for step in range(steps):
        optimiser.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimiser.step()
        if step == 0:
            first = float(loss.item())
        last = float(loss.item())

    model.eval()
    with torch.no_grad():
        accuracy = float((model(inputs).argmax(dim=1) == targets).float().mean().item())
    return {"initial_loss": first, "final_loss": last, "train_accuracy": accuracy}


def run_smoke(*, out_dir: str | Path = "outputs/smoke", device: str = "cpu", quick: bool = True) -> int:
    """Generate a cohort, run the whole pipeline over it, and report."""
    from sleepstatelab.baselines.classical import run_baselines
    from sleepstatelab.data.prepare import load_cached, prepare, reject_mask_flags
    from sleepstatelab.data.preprocess import bandpass
    from sleepstatelab.data.splits import grouped_split
    from sleepstatelab.devices import resolve
    from sleepstatelab.evaluation.metrics import evaluate_predictions
    from sleepstatelab.evaluation.predictions import PredictionWriter
    from sleepstatelab.evaluation.report import build_report
    from sleepstatelab.features.spectral import feature_matrix
    from sleepstatelab.synthetic import make_cohort
    from sleepstatelab.training.checkpoint import load_checkpoint
    from sleepstatelab.training.dataset import build_datasets
    from sleepstatelab.training.trainer import predict, train_d1, train_d2
    from sleepstatelab.training.windows import build_window_datasets, shuffle_context

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    resolved = resolve(device)
    print(f"SYNTHETIC smoke run on {resolved}. Nothing here is a result.\n")

    with tempfile.TemporaryDirectory() as scratch:
        data_root = Path(scratch) / "sleep-edf"
        cohort = make_cohort(data_root, n_participants=6, n_epochs=40 if quick else 120)
        print(f"1. generated {len(cohort)} synthetic night(s) as EDF under {data_root}")

        config = Config(
            name="synthetic-smoke",
            data=DataConfig(root=str(data_root), cache_dir=str(Path(scratch) / "cache")),
            model=ModelConfig(embedding_dim=64, block_channels=(32, 48, 64)),
            train=TrainConfig(
                device=resolved,
                epochs=4 if quick else 12,
                batch_size=32,
                early_stopping_patience=4,
            ),
        )

        report = prepare(config, progress=False)
        print(f"2. prepared: {report.stored_epochs} epochs, {report.eligible_epochs} eligible")
        print(f"   exclusions: {json.dumps(report.exclusions)}")

        records = load_cached(config)
        gaps = sum(int(np.count_nonzero(np.diff(r.epoch_index) > 1)) for r in records)
        print(f"   deliberate gaps preserved in the epoch indices: {gaps}")

        participants = sorted({r.participant_id for r in records})
        split = grouped_split(participants, seed=config.split.seed, name="synthetic")
        split.write(target / "split.json")
        print(f"3. {split.summary()}")

        train, val, test, stats = build_datasets(config, split)
        print(
            f"4. train {len(train)} / val {len(val)} / test {len(test)} epochs; "
            f"normalization {stats.method} fitted on {list(stats.fitted_on)}"
        )
        overlap = set(stats.fitted_on) & set(split.test)
        if overlap:
            raise AssertionError(f"normalization saw test participants {sorted(overlap)}")

        _, checkpoint, _ = train_d1(
            config,
            split,
            train,
            val,
            device=resolved,
            checkpoint_path=target / "d1_synthetic.pt",
            run_id="synthetic-smoke",
            progress=True,
        )
        print(f"5. trained D1: {checkpoint.notes['n_parameters']}")

        reloaded, loaded = load_checkpoint(
            target / "d1_synthetic.pt",
            expect_channels=tuple(config.data.channels),
            expect_preprocessing_id=config.preprocessing_identity,
        )
        probabilities = predict(reloaded.to(resolved), test, device=resolved)
        print(f"6. checkpoint reloaded; split id round-trip {loaded.split_id == split.identity}")

        # D2: the same encoder under a transformer over eleven epochs. Trained
        # here only to show the temporal path runs end to end on a machine with
        # no data; five passes over generated signals is not an experiment.
        window_train, window_val, window_test, _ = build_window_datasets(
            config, split, context=config.model.context_epochs
        )
        print(
            f"6b. windows: train {len(window_train)} / val {len(window_val)} / "
            f"test {len(window_test)}; genuine-neighbour coverage "
            f"{window_train.context_coverage():.3f}"
        )
        _, d2_checkpoint, _ = train_d2(
            config,
            split,
            window_train,
            window_val,
            device=resolved,
            checkpoint_path=target / "d2_synthetic.pt",
            run_id="synthetic-smoke-d2",
            progress=True,
        )
        d2_model, d2_loaded = load_checkpoint(
            target / "d2_synthetic.pt",
            expect_channels=tuple(config.data.channels),
            expect_preprocessing_id=config.preprocessing_identity,
        )
        d2_probabilities = predict(d2_model.to(resolved), window_test, device=resolved)
        print(
            f"6c. D2 trained and reloaded: {d2_checkpoint.notes['n_parameters']}; "
            f"context {d2_loaded.temporal_kwargs.get('context')} epochs"
        )

        # The control that says whether the context is being used at all.
        shuffle_context(window_test, seed=1)
        shuffled_probabilities = predict(d2_model, window_test, device=resolved)
        moved = float(np.mean(np.abs(d2_probabilities - shuffled_probabilities)))
        print(
            f"6d. shuffled-neighbour control: mean absolute change in probability "
            f"{moved:.4f}"
        )

        reject = reject_mask_flags(tuple(config.preprocess.qc_reject))

        def features_for(participants: set[str]) -> np.ndarray:
            """The same epochs and the same band-pass the CNN was given."""
            block = np.concatenate(
                [r.signals[r.eligible(reject)] for r in records if r.participant_id in participants]
            )
            return feature_matrix(
                bandpass(block, config.data.sampling_rate_hz, config.preprocess),
                config.data.sampling_rate_hz,
            )

        train_x = features_for(set(split.train))
        train_y = np.concatenate(
            [r.labels[r.eligible(reject)] for r in records if r.participant_id in split.train]
        ).astype(int)
        test_x = features_for(set(split.test))
        baselines = run_baselines(train_x=train_x, train_y=train_y, eval_x=test_x)
        print(f"7. baselines fitted: {', '.join(baselines)}")

        predictions_path = target / "predictions_synthetic.csv"
        with PredictionWriter(predictions_path) as writer:
            writer.write(
                run_id="synthetic-smoke",
                model="D1",
                split_id=split.identity,
                split_part="test",
                seed=config.train.seed,
                participant_ids=[e.participant_id for e in test.entries],
                recording_ids=[e.recording_id for e in test.entries],
                epoch_indices=np.array([e.epoch_index for e in test.entries]),
                true_labels=test.y,
                probabilities=probabilities,
                qc_flags=np.array([e.qc_flags for e in test.entries]),
            )
            writer.write(
                run_id="synthetic-smoke",
                model="D2",
                split_id=split.identity,
                split_part="test",
                seed=config.train.seed,
                participant_ids=[e.participant_id for e in window_test.entries],
                recording_ids=[e.recording_id for e in window_test.entries],
                epoch_indices=np.array([e.epoch_index for e in window_test.entries]),
                true_labels=window_test.y,
                probabilities=d2_probabilities,
                qc_flags=np.array([e.qc_flags for e in window_test.entries]),
            )
            for name, block in baselines.items():
                writer.write(
                    run_id="synthetic-smoke",
                    model=name,
                    split_id=split.identity,
                    split_part="test",
                    seed=config.split.seed,
                    participant_ids=[e.participant_id for e in test.entries],
                    recording_ids=[e.recording_id for e in test.entries],
                    epoch_indices=np.array([e.epoch_index for e in test.entries]),
                    true_labels=test.y,
                    probabilities=block,
                    qc_flags=np.array([e.qc_flags for e in test.entries]),
                )
        print(f"8. predictions saved to {predictions_path}")

    results = evaluate_predictions(predictions_path, split_part="test")
    for result in results:
        print(f"   [SYNTHETIC] {result.summary()}")
    text = build_report(
        predictions_path,
        split_part="test",
        title="SYNTHETIC smoke run - not a result",
    )
    (target / "report_synthetic.md").write_text(text)
    print(f"9. report written to {target / 'report_synthetic.md'}")

    check = overfit_check(device=resolved)
    print(
        f"10. overfit check: loss {check['initial_loss']:.3f} -> {check['final_loss']:.4f}, "
        f"train accuracy {check['train_accuracy']:.3f}"
    )
    _ = dataclasses
    if check["train_accuracy"] < 1.0 or check["final_loss"] > 0.05:
        print("    FAILED: D1 could not memorise a tiny batch; the gradient path is suspect")
        return 1
    print("\nSynthetic smoke run complete. None of these numbers describe real sleep.")
    return 0
