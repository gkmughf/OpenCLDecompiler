# ruff: noqa: TRY003, TRY400
from __future__ import annotations

import argparse
import importlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


LOGGER = logging.getLogger("cl_to_ptx")
DEFAULT_TESTS_DIR = Path("tests")

os.environ.setdefault("PYOPENCL_COMPILER_OUTPUT", "1")


class CompileError(Exception):
    pass


def get_opencl() -> Any:
    try:
        return importlib.import_module("pyopencl")
    except ModuleNotFoundError as exc:
        raise CompileError(
            "pyopencl is not installed. Install it or run this script with Python that has pyopencl."
        ) from exc


def create_nvidia_context() -> Any:
    opencl = get_opencl()

    try:
        platforms = opencl.get_platforms()
    except opencl.Error as exc:
        raise CompileError(f"Cannot query OpenCL platforms: {exc}") from exc

    for platform in platforms:
        if "NVIDIA" not in platform.name.upper():
            continue

        devices = platform.get_devices(device_type=opencl.device_type.GPU)
        if not devices:
            continue

        LOGGER.info("[NVIDIA] Using platform '%s', device '%s'", platform.name, devices[0].name)
        return opencl.Context(devices=devices)

    raise CompileError("NVIDIA GPU platform not found in PyOpenCL.")


def find_cl_files(root: Path) -> list[Path]:
    if not root.exists():
        raise CompileError(f"Input path '{root}' not found.")

    if root.is_file():
        if root.suffix.lower() != ".cl":
            raise CompileError(f"Input file '{root}' is not a .cl file.")
        return [root]

    return sorted(path for path in root.rglob("*.cl") if path.is_file())


def compile_to_ptx(
    input_file: str | Path,
    output_file: str | Path | None = None,
    *,
    ctx: Any | None = None,
    build_options: Iterable[str] = (),
) -> Path:
    opencl = get_opencl()
    input_path = Path(input_file)
    output_path = Path(output_file) if output_file is not None else input_path.with_suffix(".ptx")

    if not input_path.exists():
        raise CompileError(f"Input file '{input_path}' not found.")

    if input_path.suffix.lower() != ".cl":
        raise CompileError(f"Input file '{input_path}' is not a .cl file.")

    try:
        source = input_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CompileError(f"Cannot read '{input_path}': {exc}") from exc

    context = ctx or create_nvidia_context()

    LOGGER.info("[NVIDIA] Compiling '%s' -> '%s'", input_path, output_path)

    try:
        program = opencl.Program(context, source).build(options=list(build_options))
    except opencl.RuntimeError as exc:
        raise CompileError(f"Compilation failed for '{input_path}':\n{exc}") from exc

    binaries = program.get_info(opencl.program_info.BINARIES)
    if not binaries or not binaries[0]:
        raise CompileError(f"No binary returned by driver for '{input_path}'.")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(binaries[0])
    except OSError as exc:
        raise CompileError(f"Cannot write '{output_path}': {exc}") from exc

    LOGGER.info("[NVIDIA] Saved '%s'", output_path)
    return output_path


def compile_many(
    source_files: Iterable[Path],
    *,
    ctx: Any,
    build_options: Iterable[str],
    fail_fast: bool,
    skip_existing: bool,
) -> tuple[int, int, int]:
    compiled = 0
    failed = 0
    skipped = 0

    for source_file in source_files:
        output_file = source_file.with_suffix(".ptx")

        if skip_existing and output_file.exists():
            LOGGER.debug("[NVIDIA] Skipping existing '%s'", output_file)
            skipped += 1
            continue

        try:
            compile_to_ptx(source_file, output_file, ctx=ctx, build_options=build_options)
        except CompileError as exc:
            failed += 1
            LOGGER.error("%s", exc)
            if fail_fast:
                raise
        else:
            compiled += 1

    return compiled, failed, skipped


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile OpenCL .cl files to NVIDIA PTX.")
    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_TESTS_DIR,
        help="Input .cl file or directory with .cl files. Defaults to tests/.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output .ptx file. Only valid when input is a single .cl file.",
    )
    parser.add_argument(
        "--build-option",
        action="append",
        default=[],
        help="Extra OpenCL build option. Can be passed more than once.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop batch compilation after the first failed .cl file.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not overwrite .ptx files that already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    input_path = Path(args.input)

    if args.output and not input_path.is_file():
        LOGGER.error("Error: positional output is only valid when input is a single .cl file.")
        return 2

    try:
        source_files = find_cl_files(input_path)
        if not source_files:
            LOGGER.warning("No .cl files found under '%s'.", input_path)
            return 0

        ctx = create_nvidia_context()

        if input_path.is_file():
            output_path = Path(args.output) if args.output else input_path.with_suffix(".ptx")
            compile_to_ptx(input_path, output_path, ctx=ctx, build_options=args.build_option)
            return 0

        LOGGER.info("Found %d .cl file(s) under '%s'.", len(source_files), input_path)
        compiled, failed, skipped = compile_many(
            source_files,
            ctx=ctx,
            build_options=args.build_option,
            fail_fast=args.fail_fast,
            skip_existing=args.skip_existing,
        )
    except CompileError as exc:
        LOGGER.error("Error: %s", exc)
        return 1

    LOGGER.info("Done: %d compiled, %d failed, %d skipped.", compiled, failed, skipped)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
