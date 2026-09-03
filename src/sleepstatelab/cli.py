"""The command line: one subcommand per stage, each a thin wrapper over a module.

Nothing is implemented here. Every command below builds a configuration, calls
into the package, and prints what came back -- so anything the command line can
do is available from a notebook or a script on the same terms, and the pipeline
does not live in an argument parser.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from sleepstatelab import __version__
from sleepstatelab.config import Config, load
from sleepstatelab.devices import probe, resolve
from sleepstatelab.labels import STAGES


def _with_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Apply the handful of overrides the command line accepts.

    Deliberately few. Anything that changes what a model sees belongs in a
    configuration file that can be hashed and recorded, not in a shell history.
    """
    import dataclasses

    data = config.data
    train = config.train
    if getattr(args, "data_root", None):
        data = dataclasses.replace(data, root=args.data_root)
    if getattr(args, "cache_dir", None):
        data = dataclasses.replace(data, cache_dir=args.cache_dir)
    if getattr(args, "checksums", None):
        data = dataclasses.replace(data, checksums=args.checksums)
    if getattr(args, "participants", None):
        data = dataclasses.replace(data, participants=tuple(args.participants))
    if getattr(args, "device", None):
        train = dataclasses.replace(train, device=args.device)
    if getattr(args, "epochs", None):
        train = dataclasses.replace(train, epochs=int(args.epochs))
    if getattr(args, "seed", None) is not None:
        train = dataclasses.replace(train, seed=int(args.seed))
    return dataclasses.replace(config, data=data, train=train)


def cmd_doctor(args: argparse.Namespace) -> int:
    """What this machine can run, and what is installed."""
    print(f"sleepstatelab {__version__}")
    print(probe().summary())
    print(f"resolved device for --device {args.device}: {resolve(args.device)}")
    for name in ("numpy", "scipy", "sklearn", "torch", "mne", "matplotlib"):
        try:
            module = __import__(name)
            print(f"{name}: {getattr(module, '__version__', 'unknown')}")
        except ImportError:
            print(f"{name}: NOT INSTALLED")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Discover recordings and write the manifest."""
    from sleepstatelab.data.manifest import build_manifest

    config = _with_overrides(load(args.config), args)
    started = time.time()
    manifest = build_manifest(
        config.data.root,
        channels=tuple(config.data.channels),
        epoch_seconds=config.data.epoch_seconds,
        expected_rate_hz=config.data.sampling_rate_hz,
        participants=tuple(config.data.participants),
        nights=tuple(config.data.nights),
        checksum_file=config.data.checksums or None,
        checksums=not args.no_checksums,
        progress=True,
    )
    manifest.write(args.output)
    print()
    print(manifest.summary())
    print(f"participants: {', '.join(manifest.participants)}")
    unusable = [e for e in manifest.entries if not e.usable]
    if unusable:
        print(f"\n{len(unusable)} recording(s) with problems:")
        for entry in unusable:
            print(f"  {entry.recording_id}: {'; '.join(entry.problems)}")
    print(f"\nwritten to {args.output} in {time.time() - started:.0f}s")
    print(
        "This is what is present in the data root. Sleep Cassette holds 153 "
        "recordings from 78 participants; a smaller manifest is a subset."
    )
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    """Cut every discovered recording into epochs and cache them."""
    from sleepstatelab.data.prepare import prepare

    config = _with_overrides(load(args.config), args)
    started = time.time()
    report = prepare(config, progress=True, force=args.force)
    print()
    print(report.summary())
    print(f"exclusions: {json.dumps(report.exclusions)}")
    print(f"cache: {report.cache_dir}  ({time.time() - started:.0f}s)")
    return 0


def cmd_audit_report(args: argparse.Namespace) -> int:
    """The single-recording audit, with figures."""
    from sleepstatelab.data.audit import audit_report

    config = _with_overrides(load(args.config), args)
    _, path = audit_report(config, args.recording, out_dir=args.out_dir)
    print(f"written to {path}")
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    """Generate the participant-disjoint split."""
    from sleepstatelab.data.prepare import load_cached
    from sleepstatelab.data.splits import grouped_split, label_budget_subsets

    config = _with_overrides(load(args.config), args)
    participants = sorted({record.participant_id for record in load_cached(config)})
    split = grouped_split(
        participants,
        seed=config.split.seed,
        train_fraction=config.split.train_fraction,
        val_fraction=config.split.val_fraction,
        name=args.name,
    )
    split.write(args.output)
    print(split.summary())
    print(f"train: {', '.join(split.train)}")
    print(f"val:   {', '.join(split.val)}")
    print(f"test:  {', '.join(split.test)}")
    budgets = label_budget_subsets(split, tuple(config.split.label_budgets), seed=config.split.seed)
    print("\nnested label budgets over the training participants:")
    for budget, people in budgets.items():
        print(f"  {budget:>5.0%}: {len(people)} participant(s) -- {', '.join(people)}")
    print(f"\nwritten to {args.output}")
    return 0


def cmd_baselines(args: argparse.Namespace) -> int:
    """Fit the classical baselines and save their predictions."""
    from sleepstatelab.baselines.classical import run_baselines
    from sleepstatelab.data.prepare import load_cached, reject_mask_flags
    from sleepstatelab.data.preprocess import bandpass
    from sleepstatelab.data.splits import Split
    from sleepstatelab.evaluation.predictions import PredictionWriter
    from sleepstatelab.features.spectral import feature_matrix
    from sleepstatelab.provenance import make_run_provenance

    config = _with_overrides(load(args.config), args)
    split = Split.read(args.split)
    reject = reject_mask_flags(tuple(config.preprocess.qc_reject))

    def block(participants: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, list, list, np.ndarray, np.ndarray]:
        records = load_cached(config, participants)
        features, labels, people, recordings, indices, flags = [], [], [], [], [], []
        for record in records:
            keep = record.eligible(reject)
            if not keep.any():
                continue
            print(f"  features for {record.recording_id} ({int(keep.sum())} epochs)", flush=True)
            # The same band-pass the CNN's inputs get, so the two models are
            # given the same signal and differ only in what they do with it.
            # Normalisation is deliberately not applied: the features are
            # scale-aware by design and the scaler inside each pipeline is
            # fitted on the training fold anyway.
            filtered = bandpass(record.signals[keep], record.sampling_rate_hz, config.preprocess)
            features.append(feature_matrix(filtered, record.sampling_rate_hz))
            labels.append(record.labels[keep].astype(int))
            people.extend([record.participant_id] * int(keep.sum()))
            recordings.extend([record.recording_id] * int(keep.sum()))
            indices.append(record.epoch_index[keep])
            flags.append(record.qc[keep])
        return (
            np.vstack(features),
            np.concatenate(labels),
            people,
            recordings,
            np.concatenate(indices),
            np.concatenate(flags),
        )

    print("training features:")
    train_x, train_y, _, _, _, _ = block(split.train)
    print(f"evaluation features ({args.part}):")
    evaluate_on = getattr(split, args.part)
    eval_x, eval_y, people, recordings, indices, flags = block(evaluate_on)

    run_id = args.run_id or f"baselines-{split.identity}"
    predictions = run_baselines(train_x=train_x, train_y=train_y, eval_x=eval_x)
    with PredictionWriter(args.output) as writer:
        for name, probabilities in predictions.items():
            writer.write(
                run_id=run_id,
                model=name,
                split_id=split.identity,
                split_part=args.part,
                seed=config.split.seed,
                participant_ids=people,
                recording_ids=recordings,
                epoch_indices=indices,
                true_labels=eval_y,
                probabilities=probabilities,
                qc_flags=flags,
            )
    provenance = make_run_provenance(
        run_id=run_id,
        device="cpu",
        seed=config.split.seed,
        config=config.to_dict(),
        split_id=split.identity,
        channels=tuple(config.data.channels),
        label_order=STAGES,
        preprocessing_id=config.preprocessing_identity,
        notes=f"classical baselines on {args.part}",
        extra={"n_train_epochs": int(train_x.shape[0]), "n_features": int(train_x.shape[1])},
    )
    provenance.write(Path(args.output).with_suffix(".provenance.json"))
    print(f"\n{train_x.shape[0]} training epochs, {eval_x.shape[0]} evaluation epochs")
    print(f"predictions written to {args.output}")
    return 0


def cmd_train_d1(args: argparse.Namespace) -> int:
    """Train D1 and save its checkpoint."""
    from sleepstatelab.data.splits import Split, label_budget_subsets
    from sleepstatelab.provenance import make_run_provenance
    from sleepstatelab.training.dataset import build_datasets
    from sleepstatelab.training.trainer import train_d1

    config = _with_overrides(load(args.config), args)
    device = resolve(config.train.device)
    split = Split.read(args.split)

    budget_participants = None
    if args.label_budget is not None:
        budget_participants = label_budget_subsets(
            split, (args.label_budget,), seed=config.split.seed
        )[args.label_budget]
        print(
            f"label budget {args.label_budget:.0%}: training on "
            f"{len(budget_participants)} of {len(split.train)} participants. "
            "Validation labels are NOT reduced."
        )

    train, val, test, stats = build_datasets(
        config, split, train_participants=budget_participants
    )
    print(
        f"train {len(train)} epochs / {len(train.participants)} participants; "
        f"val {len(val)} / {len(val.participants)}; "
        f"test {len(test)} / {len(test.participants)}"
    )
    print(f"normalization {stats.method} [{stats.identity}] fitted on {list(stats.fitted_on)}")
    print(f"class counts (train): {dict(zip(STAGES, train.class_counts().tolist(), strict=True))}")
    print(f"class weights (train): {np.round(train.class_weights(config.train.class_weighting), 3).tolist()}")
    print(f"device: {device}")

    run_id = args.run_id or f"d1-{split.identity}-s{config.train.seed}"
    _, checkpoint, _ = train_d1(
        config,
        split,
        train,
        val,
        device=device,
        checkpoint_path=args.checkpoint,
        run_id=run_id,
    )
    print(
        f"\nselected epoch {checkpoint.epoch_selected + 1} with validation "
        f"participant macro-F1 {checkpoint.val_metric_value:.4f}"
    )
    print(f"parameters: {checkpoint.notes['n_parameters']}")
    print(f"checkpoint written to {args.checkpoint}")
    make_run_provenance(
        run_id=run_id,
        device=device,
        seed=config.train.seed,
        config=config.to_dict(),
        split_id=split.identity,
        channels=tuple(config.data.channels),
        label_order=STAGES,
        preprocessing_id=config.preprocessing_identity,
        notes="D1 supervised training",
        extra={"label_budget": args.label_budget, "checkpoint": str(args.checkpoint)},
    ).write(Path(args.checkpoint).with_suffix(".provenance.json"))
    return 0


def cmd_train_d2(args: argparse.Namespace) -> int:
    """Train D2: the same encoder, with a transformer over eleven epochs."""
    from sleepstatelab.data.splits import Split, label_budget_subsets
    from sleepstatelab.provenance import make_run_provenance
    from sleepstatelab.training.checkpoint import load_checkpoint
    from sleepstatelab.training.trainer import train_d2
    from sleepstatelab.training.windows import build_window_datasets

    config = _with_overrides(load(args.config), args)
    device = resolve(config.train.device)
    split = Split.read(args.split)

    budget_participants = None
    if args.label_budget is not None:
        budget_participants = label_budget_subsets(
            split, (args.label_budget,), seed=config.split.seed
        )[args.label_budget]
        print(
            f"label budget {args.label_budget:.0%}: training on "
            f"{len(budget_participants)} of {len(split.train)} participants. "
            "Validation labels are NOT reduced."
        )

    encoder = None
    initialised_from = None
    if args.init_encoder:
        # The route a pretrained backbone takes into D2 -- this is D3. What is
        # loaded is named in the checkpoint, so a run started from a pretrained
        # encoder can never be mistaken for one started from random weights.
        from sleepstatelab.training.checkpoint import (
            is_encoder_checkpoint,
            load_encoder_checkpoint,
        )

        if is_encoder_checkpoint(args.init_encoder):
            encoder, source_encoder = load_encoder_checkpoint(
                args.init_encoder,
                expect_channels=tuple(config.data.channels),
                expect_preprocessing_id=config.preprocessing_identity,
                forbid_participants=tuple(split.val) + tuple(split.test),
            )
            initialised_from = {
                "path": str(args.init_encoder),
                "kind": "self-supervised encoder",
                "objective": source_encoder.objective,
                "pretrain_participants": list(source_encoder.pretrain_participants),
                "run_id": source_encoder.notes.get("run_id", "unknown"),
            }
            print(
                f"encoder initialised from {args.init_encoder}: "
                f"{source_encoder.objective}, pretrained on "
                f"{', '.join(source_encoder.pretrain_participants)}"
            )
        else:
            source, source_checkpoint = load_checkpoint(
                args.init_encoder,
                expect_channels=tuple(config.data.channels),
                expect_preprocessing_id=config.preprocessing_identity,
            )
            encoder = source.encoder
            initialised_from = {
                "path": str(args.init_encoder),
                "kind": f"supervised {source_checkpoint.model_name}",
                "run_id": source_checkpoint.notes.get("run_id", "unknown"),
            }
            print(
                f"encoder initialised from {args.init_encoder} "
                f"({source_checkpoint.model_name}, run "
                f"{source_checkpoint.notes.get('run_id', 'unknown')})"
            )

    segments = not args.no_segments
    centres = config.model.centres_per_segment
    train, val, test, stats = build_window_datasets(
        config,
        split,
        context=config.model.context_epochs,
        train_participants=budget_participants,
        segments=segments,
        centres_per_segment=centres,
    )
    if segments:
        print(
            f"train {train.n_centres()} centres in {len(train)} segments of "
            f"{centres} / {len(train.participants)} participants; "
            f"val {val.n_centres()} / {len(val.participants)}; "
            f"test {test.n_centres()} / {len(test.participants)}"
        )
        naive = train.n_centres() * config.model.context_epochs
        print(
            f"encodings per pass: {train.encodings_per_pass()} against {naive} "
            f"window-by-window ({naive / train.encodings_per_pass():.1f}x less)"
        )
    else:
        print(
            f"train {len(train)} windows / {len(train.participants)} participants; "
            f"val {len(val)} / {len(val.participants)}; "
            f"test {len(test)} / {len(test.participants)}"
        )
    print(
        f"context {config.model.context_epochs} epochs; genuine-neighbour "
        f"coverage: train {train.context_coverage():.3f}, "
        f"val {val.context_coverage():.3f}, test {test.context_coverage():.3f}"
    )
    print(f"normalization {stats.method} [{stats.identity}] fitted on {list(stats.fitted_on)}")
    print(f"class counts (train): {dict(zip(STAGES, train.class_counts().tolist(), strict=True))}")
    print(f"device: {device}")

    # train.batch_size counts examples everywhere in this package, so a segment
    # batch is however many segments make up that many centres. The
    # configuration is left alone: a checkpoint recording "batch_size: 2" would
    # be describing segments while claiming to describe examples.
    loader_batch = max(1, config.train.batch_size // centres) if segments else None
    if segments:
        print(f"batch: {loader_batch} segment(s) per step = {loader_batch * centres} centres")

    run_id = args.run_id or f"d2-{split.identity}-s{config.train.seed}"
    _, checkpoint, _ = train_d2(
        config,
        split,
        train,
        val,
        encoder=encoder,
        device=device,
        checkpoint_path=args.checkpoint,
        run_id=run_id,
        loader_batch_size=loader_batch,
    )
    print(
        f"\nselected epoch {checkpoint.epoch_selected + 1} with validation "
        f"participant macro-F1 {checkpoint.val_metric_value:.4f}"
    )
    print(f"parameters: {checkpoint.notes['n_parameters']}")
    print(f"checkpoint written to {args.checkpoint}")
    make_run_provenance(
        run_id=run_id,
        device=device,
        seed=config.train.seed,
        config=config.to_dict(),
        split_id=split.identity,
        channels=tuple(config.data.channels),
        label_order=STAGES,
        preprocessing_id=config.preprocessing_identity,
        notes="D2 supervised training (offline: uses future context)",
        extra={
            "label_budget": args.label_budget,
            "checkpoint": str(args.checkpoint),
            "init_encoder": initialised_from,
            "context_epochs": config.model.context_epochs,
            "context_coverage_train": train.context_coverage(),
            "shared_encodings": segments,
            "centres_per_segment": centres if segments else None,
        },
    ).write(Path(args.checkpoint).with_suffix(".provenance.json"))
    return 0


def cmd_pretrain(args: argparse.Namespace) -> int:
    """Self-supervised masked reconstruction on training participants only."""
    from sleepstatelab.data.splits import Split
    from sleepstatelab.provenance import make_run_provenance
    from sleepstatelab.training.dataset import build_datasets
    from sleepstatelab.training.pretrain import pretrain_encoder

    config = _with_overrides(load(args.config), args)
    device = resolve(config.train.device)
    split = Split.read(args.split)

    train, val, _, stats = build_datasets(config, split)
    print(
        f"pretraining on {len(train)} epochs from {len(train.participants)} "
        f"training participant(s): {', '.join(train.participants)}"
    )
    print(
        f"held out of pretraining: val {', '.join(split.val)}; "
        f"test {', '.join(split.test)}"
    )
    # The patch is derived from the encoder unless it was set by hand, so the
    # model is the authority on what is about to happen -- not the config file.
    from sleepstatelab.training.pretrain import build_pretrainer

    shape = build_pretrainer(config)
    print(
        f"masking {shape.n_patches_masked} of {shape.n_patches} patches of "
        f"{shape.patch_samples} samples ({shape.mask_ratio:.0%}); "
        f"tokens cover {shape.covered_samples} of {config.samples_per_epoch} "
        "samples; no labels are read"
    )
    print(f"normalization {stats.method} [{stats.identity}]")
    print(f"device: {device}")

    run_id = args.run_id or f"pretrain-{split.identity}-s{config.train.seed}"
    _, checkpoint, _ = pretrain_encoder(
        config,
        split,
        train,
        val,
        device=device,
        checkpoint_path=args.checkpoint,
        run_id=run_id,
    )
    print(
        f"\nselected epoch {checkpoint.epoch_selected + 1} with validation "
        f"reconstruction loss {checkpoint.metric_value:.4f}"
    )
    print(f"parameters: {checkpoint.notes['n_parameters']}")
    print(f"encoder written to {args.checkpoint}")
    make_run_provenance(
        run_id=run_id,
        device=device,
        seed=config.train.seed,
        config=config.to_dict(),
        split_id=split.identity,
        channels=tuple(config.data.channels),
        label_order=STAGES,
        preprocessing_id=config.preprocessing_identity,
        notes="self-supervised masked reconstruction; no labels read",
        extra={
            "objective": checkpoint.objective,
            "pretrain_participants": list(checkpoint.pretrain_participants),
            "checkpoint": str(args.checkpoint),
        },
    ).write(Path(args.checkpoint).with_suffix(".provenance.json"))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Run a checkpoint over a split part and save one row per epoch."""
    from sleepstatelab.data.preprocess import NormalizationStats
    from sleepstatelab.data.splits import Split
    from sleepstatelab.evaluation.predictions import PredictionWriter
    from sleepstatelab.training.checkpoint import load_checkpoint
    from sleepstatelab.training.dataset import build_datasets
    from sleepstatelab.training.trainer import predict

    config = _with_overrides(load(args.config), args)
    device = resolve(config.train.device)
    split = Split.read(args.split)
    model, checkpoint = load_checkpoint(
        args.checkpoint,
        expect_channels=tuple(config.data.channels),
        expect_preprocessing_id=config.preprocessing_identity,
    )
    if checkpoint.split_id != split.identity:
        raise SystemExit(
            f"checkpoint was trained on split {checkpoint.split_id} and "
            f"{args.split} is {split.identity}. Refusing: the test participants "
            "may not be the same people."
        )
    model = model.to(device)

    # The checkpoint's own normalisation, not a freshly fitted one: the model
    # must see signals scaled exactly as they were during training.
    stats = NormalizationStats(
        method=checkpoint.normalization["method"],
        channels=tuple(checkpoint.normalization["channels"]),
        centre=tuple(checkpoint.normalization["centre"]),
        scale=tuple(checkpoint.normalization["scale"]),
        fitted_on=tuple(checkpoint.normalization["fitted_on"]),
        n_epochs=checkpoint.normalization["n_epochs"],
        clip_sigma=checkpoint.normalization["clip_sigma"],
    )
    leaked = set(stats.fitted_on) & set(split.test)
    if leaked:
        raise SystemExit(
            f"refusing to predict: the checkpoint's normalization was fitted on "
            f"test participants {sorted(leaked)}"
        )
    if checkpoint.temporal_kwargs:
        from sleepstatelab.training.windows import build_window_datasets

        train, val, test, _ = build_window_datasets(
            config,
            split,
            context=int(checkpoint.temporal_kwargs.get("context", 11)),
            stats=stats,
        )
    else:
        train, val, test, _ = build_datasets(config, split, stats=stats)
    dataset = {"train": train, "val": val, "test": test}[args.part]
    if (args.shuffle_context or args.mask_context) and not checkpoint.temporal_kwargs:
        raise SystemExit("the context controls only mean something for a temporal model")
    if args.shuffle_context and args.mask_context:
        raise SystemExit(
            "--shuffle-context and --mask-context are different questions; run them "
            "separately so each result says which was asked"
        )
    if args.shuffle_context:
        from sleepstatelab.training.windows import shuffle_context

        shuffle_context(dataset, seed=args.shuffle_seed)
        print(
            f"CONTROL RUN: the non-central positions of every window have been "
            f"shuffled with seed {args.shuffle_seed}. If the score holds up, the "
            "model is not using the ORDER of its context -- which is not the same "
            "as not using the context."
        )
    if args.mask_context:
        from sleepstatelab.training.windows import mask_context

        mask_context(dataset)
        print(
            "CONTROL RUN: every non-central position has been marked absent, so "
            "the model is reduced to its encoder on the central epoch. If the "
            "score holds up, the context was contributing nothing at all."
        )
    probabilities = predict(model, dataset, device=device)
    with PredictionWriter(args.output, overwrite=not args.append) as writer:
        writer.write(
            run_id=checkpoint.notes.get("run_id", "d1"),
            model=args.model_name,
            split_id=split.identity,
            split_part=args.part,
            seed=checkpoint.seed,
            participant_ids=[e.participant_id for e in dataset.entries],
            recording_ids=[e.recording_id for e in dataset.entries],
            epoch_indices=np.array([e.epoch_index for e in dataset.entries]),
            true_labels=dataset.y,
            probabilities=probabilities,
            qc_flags=np.array([e.qc_flags for e in dataset.entries]),
        )
    print(f"{len(dataset)} predictions written to {args.output}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Build the report tables from saved predictions."""
    from sleepstatelab.evaluation.metrics import evaluate_predictions
    from sleepstatelab.evaluation.report import build_report

    results = evaluate_predictions(args.predictions, split_part=args.part)
    for result in results:
        print(result.summary())
    text = build_report(args.predictions, split_part=args.part, title=args.title)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text)
    if args.json:
        Path(args.json).write_text(
            json.dumps([r.to_dict() for r in results], indent=2, default=str)
        )
    print(f"\nreport written to {args.output}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """The synthetic end-to-end run: no recordings, CPU only."""
    from sleepstatelab.smoke import run_smoke

    return run_smoke(out_dir=args.out_dir, device=args.device, quick=not args.full)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sleepstatelab",
        description=(
            "SleepStateLab: self-supervised EEG representation learning and "
            "sleep-state decoding. Every subcommand is a wrapper over an "
            "importable module."
        ),
    )
    parser.add_argument("--version", action="version", version=f"sleepstatelab {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--config", help="path to a YAML configuration file")
        sub.add_argument("--data-root", help="override the configured data root")
        sub.add_argument("--cache-dir", help="override the configured epoch cache")
        sub.add_argument("--participants", nargs="*", help="restrict to these participants")

    doctor = subparsers.add_parser("doctor", help="what this machine can run")
    doctor.add_argument("--device", default="auto")
    doctor.set_defaults(func=cmd_doctor)

    audit = subparsers.add_parser("audit", help="discover recordings and write the manifest")
    common(audit)
    audit.add_argument("--checksums", help="path to a PhysioNet SHA256SUMS.txt")
    audit.add_argument("--no-checksums", action="store_true", help="skip hashing the files")
    audit.add_argument("--output", default="outputs/manifest.json")
    audit.set_defaults(func=cmd_audit)

    prepare = subparsers.add_parser("prepare", help="epoch every recording into the cache")
    common(prepare)
    prepare.add_argument("--force", action="store_true", help="re-epoch even if cached")
    prepare.set_defaults(func=cmd_prepare)

    report_one = subparsers.add_parser(
        "audit-report", help="the single-recording audit, with figures"
    )
    common(report_one)
    report_one.add_argument("--recording", help="recording id, default the first found")
    report_one.add_argument("--checksums", help="path to a PhysioNet SHA256SUMS.txt")
    report_one.add_argument("--out-dir", default="outputs/audit")
    report_one.set_defaults(func=cmd_audit_report)

    split = subparsers.add_parser("split", help="generate the participant-disjoint split")
    common(split)
    split.add_argument("--name", default="dev")
    split.add_argument("--output", default="outputs/split.json")
    split.set_defaults(func=cmd_split)

    baselines = subparsers.add_parser("baselines", help="fit and score the classical baselines")
    common(baselines)
    baselines.add_argument("--split", required=True)
    baselines.add_argument("--part", default="test", choices=("train", "val", "test"))
    baselines.add_argument("--output", default="outputs/predictions_baselines.csv")
    baselines.add_argument("--run-id")
    baselines.set_defaults(func=cmd_baselines)

    train = subparsers.add_parser("train-d1", help="train the epoch CNN")
    common(train)
    train.add_argument("--split", required=True)
    train.add_argument("--checkpoint", default="runs/d1/checkpoint.pt")
    train.add_argument("--device", default=None, help="cpu, cuda, mps or auto")
    train.add_argument("--epochs", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--run-id")
    train.add_argument(
        "--label-budget",
        type=float,
        help="train on a nested subset of the training participants (0-1]",
    )
    train.set_defaults(func=cmd_train_d1)

    train2 = subparsers.add_parser(
        "train-d2", help="train the temporal model over eleven epochs (offline)"
    )
    common(train2)
    train2.add_argument("--split", required=True)
    train2.add_argument("--checkpoint", default="runs/d2/checkpoint.pt")
    train2.add_argument("--device", default=None, help="cpu, cuda, mps or auto")
    train2.add_argument("--epochs", type=int)
    train2.add_argument("--seed", type=int)
    train2.add_argument("--run-id")
    train2.add_argument(
        "--init-encoder",
        help="checkpoint whose encoder initialises this one (the route D3 uses)",
    )
    train2.add_argument(
        "--label-budget",
        type=float,
        help="train on a nested subset of the training participants (0-1]",
    )
    train2.add_argument(
        "--no-segments",
        action="store_true",
        help=(
            "encode every window independently instead of reusing the encodings "
            "overlapping windows share. The same model, about eight times "
            "slower; kept so the fast path can be checked against it."
        ),
    )
    train2.set_defaults(func=cmd_train_d2)

    pre = subparsers.add_parser(
        "pretrain",
        help="self-supervised masked reconstruction on training participants only",
    )
    common(pre)
    pre.add_argument("--split", required=True)
    pre.add_argument("--checkpoint", default="runs/pretrain/encoder.pt")
    pre.add_argument("--device", default=None, help="cpu, cuda, mps or auto")
    pre.add_argument("--seed", type=int)
    pre.add_argument("--run-id")
    pre.set_defaults(func=cmd_pretrain)

    predict_cmd = subparsers.add_parser("predict", help="save one prediction row per epoch")
    common(predict_cmd)
    predict_cmd.add_argument("--split", required=True)
    predict_cmd.add_argument("--checkpoint", required=True)
    predict_cmd.add_argument("--part", default="test", choices=("train", "val", "test"))
    predict_cmd.add_argument("--output", default="outputs/predictions_d1.csv")
    predict_cmd.add_argument("--model-name", default="D1")
    predict_cmd.add_argument("--device", default=None)
    predict_cmd.add_argument("--append", action="store_true")
    predict_cmd.add_argument(
        "--shuffle-context",
        action="store_true",
        help="control: shuffle the non-central context positions before predicting",
    )
    predict_cmd.add_argument("--shuffle-seed", type=int, default=0)
    predict_cmd.add_argument(
        "--mask-context",
        action="store_true",
        help=(
            "control: mark every non-central position absent, reducing a "
            "temporal model to its encoder on the central epoch"
        ),
    )
    predict_cmd.set_defaults(func=cmd_predict)

    report = subparsers.add_parser("report", help="build tables from saved predictions")
    report.add_argument("predictions")
    report.add_argument("--part", default="test")
    report.add_argument("--output", default="outputs/report.md")
    report.add_argument("--json", help="also write the metrics as JSON")
    report.add_argument("--title", default="Results")
    report.set_defaults(func=cmd_report)

    smoke = subparsers.add_parser("smoke", help="synthetic end-to-end run, no recordings")
    smoke.add_argument("--out-dir", default="outputs/smoke")
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--full", action="store_true", help="more epochs, still synthetic")
    smoke.set_defaults(func=cmd_smoke)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
