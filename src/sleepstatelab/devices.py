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
* **Available is not the same as working.** ``torch.backends.mps.is_available()``
  answered ``True`` on a machine whose Metal shader compiler had become
  unreachable, and the first convolution then killed the process with an
  assertion rather than an exception -- taking a cohort training run with it,
  hours in. So a device is checked by *using* it, in a subprocess, because a
  failure of that kind cannot be caught in the process it happens in.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DeviceReport:
    """What this machine offers, as observed rather than as expected."""

    torch_version: str
    cpu: bool
    cuda: bool
    cuda_devices: tuple[str, ...]
    mps: bool
    mps_built: bool
    working: dict[str, bool] = field(default_factory=dict)
    """Device name to whether a small operation actually completed on it. Empty
    when the functional check was not run."""

    def note(self, device: str) -> str:
        verdict = self.working.get(device)
        if verdict is None:
            return ""
        return ", and a test operation ran" if verdict else ", but a test operation FAILED"

    def summary(self) -> str:
        lines = [f"torch {self.torch_version}", f"cpu: available{self.note('cpu')}"]
        if self.cuda:
            lines.append(
                f"cuda: available ({', '.join(self.cuda_devices)}){self.note('cuda')}"
            )
        else:
            lines.append("cuda: not available")
        if self.mps:
            lines.append(f"mps: available{self.note('mps')}")
        elif self.mps_built:
            lines.append("mps: built into torch but not available on this machine")
        else:
            lines.append("mps: not available")
        return "\n".join(lines)


#: A convolution and a backward pass: enough to make the accelerator compile and
#: run a real kernel, small enough to take milliseconds.
_TRIAL = (
    "import torch;"
    "d='{device}';"
    "x=torch.randn(4,2,600,device=d);"
    "c=torch.nn.Conv1d(2,8,9,padding=4).to(d);"
    "y=c(x).sum();"
    "y.backward();"
    "torch.mps.synchronize() if d=='mps' else None;"
    "print('ok')"
)


def device_works(device: str, *, timeout: float = 120.0) -> bool:
    """Whether a small operation actually completes on ``device``.

    Run in a subprocess on purpose. A Metal shader-compiler failure aborts the
    process rather than raising, so there is no way to ask this question safely
    from inside the process that wants the answer. The cost is one interpreter
    start, once, against the cost of losing a training run hours in.
    """
    import subprocess
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "-c", _TRIAL.format(device=device)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and "ok" in result.stdout


def probe(*, check: bool = False) -> DeviceReport:
    """Observe the devices this interpreter can actually reach.

    ``check`` additionally runs a small operation on each available device, which
    is the only way to tell an accelerator that is present from one that is
    present and functional.
    """
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
    working: dict[str, bool] = {}
    if check:
        working["cpu"] = device_works("cpu")
        if cuda:
            working["cuda"] = device_works("cuda")
        if mps:
            working["mps"] = device_works("mps")

    return DeviceReport(
        torch_version=torch.__version__,
        cpu=True,
        cuda=cuda,
        cuda_devices=names,
        mps=mps,
        mps_built=mps_built,
        working=working,
    )


def resolve(requested: str = "cpu", *, check: bool = False) -> str:
    """Turn a requested device string into one torch can be handed.

    ``auto`` prefers CUDA, then MPS, then CPU. Any other value must name a
    device that exists, or this raises: a run that asked for an accelerator and
    quietly got a CPU is a run whose timings and, through non-determinism, whose
    numbers mean something other than what the log says.
    """
    wanted = requested.strip().lower()
    report = probe(check=check)

    def refuse_if_broken(name: str) -> None:
        if report.working.get(name) is False:
            raise RuntimeError(
                f"device {name!r} reports itself available but a test operation "
                "failed on it. On Apple silicon this is usually the Metal shader "
                "compiler having become unreachable, which aborts the process "
                "rather than raising -- so a run started here would die partway "
                "through. Restarting the machine normally clears it; --device cpu "
                "works meanwhile."
            )
    if wanted == "auto":
        if report.cuda and report.working.get("cuda", True):
            return "cuda"
        if report.mps and report.working.get("mps", True):
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
        refuse_if_broken("cuda")
        return wanted
    if wanted == "mps":
        if not report.mps:
            raise RuntimeError(
                f"device {requested!r} was requested but MPS is not available here; "
                "use --device cpu or --device auto"
            )
        refuse_if_broken("mps")
        return wanted
    raise ValueError(f"unrecognised device {requested!r}; expected cpu, cuda, mps or auto")
