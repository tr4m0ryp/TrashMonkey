"""Optional read-only environment check for NVIDIA Jetson targets.

Only relevant when deploying to a Jetson-class device (L4T, power mode,
camera streams); other deployment targets skip this entirely. NEVER changes
system state -- it prints the exact remediation commands
(``sudo nvpmodel -m 2``; ``sudo jetson_clocks``) instead of running them.
Exits nonzero with a summary table when any check fails.

CLI: ``python -m yolo_waste_sorter.deploy.check_env``
"""

from __future__ import annotations

import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

NV_TEGRA_RELEASE = Path("/etc/nv_tegra_release")
MAXN_REMEDIATION = "sudo nvpmodel -m 2; sudo jetson_clocks"
CAMERA_PROBE_TIMEOUT_S = 3.0


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    remediation: str = ""


def check_l4t(release_path: Path = NV_TEGRA_RELEASE) -> CheckResult:
    """Read the L4T/JetPack identification line; fail off-Jetson."""
    if not release_path.is_file():
        return CheckResult(
            name="l4t",
            ok=False,
            detail=f"{release_path} missing -- not a Jetson (or L4T not installed)",
            remediation="flash JetPack 6.x (e.g. 6.2.2 / L4T 36.5.0)",
        )
    first_line = release_path.read_text(encoding="utf-8").splitlines()[0].strip()
    return CheckResult(name="l4t", ok=True, detail=first_line)


def check_nvpmodel(command: tuple[str, ...] = ("nvpmodel", "-q")) -> CheckResult:
    """Query the power mode; tolerate a missing nvpmodel binary."""
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except FileNotFoundError:
        return CheckResult(
            name="nvpmodel",
            ok=False,
            detail=f"{command[0]} not found -- cannot query the power mode",
            remediation=MAXN_REMEDIATION,
        )
    output = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        return CheckResult(
            name="nvpmodel",
            ok=False,
            detail=f"nvpmodel -q failed ({proc.returncode}): {output}",
            remediation=MAXN_REMEDIATION,
        )
    if "MAXN" in output.upper():
        return CheckResult(name="nvpmodel", ok=True, detail=output.replace("\n", " | "))
    return CheckResult(
        name="nvpmodel",
        ok=False,
        detail=f"power mode is not MAXN SUPER: {output.replace(chr(10), ' | ')}",
        remediation=MAXN_REMEDIATION,
    )


def check_camera(url: str, timeout_s: float = CAMERA_PROBE_TIMEOUT_S) -> CheckResult:
    """Open the stream URL with a short timeout and read the first bytes."""
    name = f"camera {url}"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310
            status = int(response.status)
            if status != 200:
                return CheckResult(
                    name=name,
                    ok=False,
                    detail=f"HTTP {status}",
                    remediation="check the camera firmware / stream endpoint",
                )
            response.read(1)  # the MJPEG stream must actually produce bytes
        return CheckResult(name=name, ok=True, detail="HTTP 200, stream produces data")
    except Exception as err:  # noqa: BLE001 -- URLError/timeout/socket all mean "down"
        return CheckResult(
            name=name,
            ok=False,
            detail=f"unreachable: {err}",
            remediation="power-cycle the camera; verify the dedicated 2.4 GHz AP (F7)",
        )


def run_checks(
    camera_urls: tuple[str, ...],
    *,
    release_path: Path = NV_TEGRA_RELEASE,
    nvpmodel_command: tuple[str, ...] = ("nvpmodel", "-q"),
    camera_timeout_s: float = CAMERA_PROBE_TIMEOUT_S,
) -> list[CheckResult]:
    results = [check_l4t(release_path), check_nvpmodel(nvpmodel_command)]
    results.extend(check_camera(url, timeout_s=camera_timeout_s) for url in camera_urls)
    return results


def format_table(results: list[CheckResult]) -> str:
    """Plain-text summary table; remediation column only where it applies."""
    rows = [("check", "status", "detail")]
    for r in results:
        detail = r.detail if r.ok or not r.remediation else f"{r.detail} -> {r.remediation}"
        rows.append((r.name, "OK" if r.ok else "FAIL", detail))
    widths = [max(len(row[i]) for row in rows) for i in range(2)]
    lines = [f"{row[0]:<{widths[0]}}  {row[1]:<{widths[1]}}  {row[2]}" for row in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m yolo_waste_sorter.deploy.check_env",
        description="Read-only Jetson deployment checks: L4T, power mode, camera streams (T8).",
    )
    parser.add_argument(
        "--config", type=Path, default=None, help="experiment yaml (default: configs/config.yaml)"
    )
    args = parser.parse_args(argv)

    from yolo_waste_sorter.utils.config import load_config

    cfg = load_config(args.config)
    results = run_checks(cfg.deploy.cameras)
    print(format_table(results))
    failed = [r for r in results if not r.ok]
    if failed:
        print(f"\n{len(failed)} check(s) failed -- remediations above; nothing was changed.")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
