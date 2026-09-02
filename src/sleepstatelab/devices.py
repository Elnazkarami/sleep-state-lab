"""Which device to compute on, decided explicitly rather than by assumption.

Two rules, both of which exist because the alternative silently changes what a
run means:

* **CPU is always available and always selectable.** ``--device cpu`` is honoured
  even on a machine with an accelerator, so a result can be reproduced on the
  machine that has nothing.
* **Nothing is assumed present.** CUDA and Apple MPS are probed, reported, and
  used only when asked for or when ``auto`` finds them. A configuration naming a
  device that is not there is an error, not a silent fall back to CPU -- falling
  back would let a benchmark run for a week on the wrong hardware and never say.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeviceReport:
    """What this machine offers, as observed rather than as expected."""

    torch_version: str
    cpu: bool
    cuda: bool
    cuda_devices: tuple[str, ...]
    mps: bool
    mps_built: bool

    def summary(self) -> str:
        lines = [f"torch {self.torch_version}", "cpu: available"]
        if self.cuda:
            lines.append(f"cuda: available ({', '.join(self.cuda_devices)})")
        else:
            lines.append("cuda: not available")
        if self.mps:
            lines.append("mps: available")
        elif self.mps_built:
            lines.append("mps: built into torch but not available on this machine")
        else:
            lines.append("mps: not available")
        return "\n".join(lines)


def probe() -> DeviceReport:
    """Observe the devices this interpreter can actually reach."""
    import torch

    cuda = bool(torch.cuda.is_available())
    names: tuple[str, ...] = ()
    if cuda:
        names = tuple(
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        )
    backend = getattr(torch.backends, "mps", None)
    mps = bool(backend is not None and backend.is_available())
    mps_built = bool(backend is not None and backend.is_built())
    return DeviceReport(
        torch_version=torch.__version__,
        cpu=True,
        cuda=cuda,
        cuda_devices=names,
        mps=mps,
        mps_built=mps_built,
    )


def resolve(requested: str = "cpu") -> str:
    """Turn a requested device string into one torch can be handed.

    ``auto`` prefers CUDA, then MPS, then CPU. Any other value must name a
    device that exists, or this raises: a run that asked for an accelerator and
    quietly got a CPU is a run whose timings and, through non-determinism, whose
    numbers mean something other than what the log says.
    """
    wanted = requested.strip().lower()
    report = probe()
    if wanted == "auto":
        if report.cuda:
            return "cuda"
        if report.mps:
            return "mps"
        return "cpu"
    if wanted == "cpu":
        return "cpu"
    if wanted.startswith("cuda"):
        if not report.cuda:
            raise RuntimeError(
                f"device {requested!r} was requested but CUDA is not available here; "
                "use --device cpu or --device auto"
            )
        return wanted
    if wanted == "mps":
        if not report.mps:
            raise RuntimeError(
                f"device {requested!r} was requested but MPS is not available here; "
                "use --device cpu or --device auto"
            )
        return wanted
    raise ValueError(f"unrecognised device {requested!r}; expected cpu, cuda, mps or auto")
