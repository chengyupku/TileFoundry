"""NVRTC/driver implementation of the local single-CTA profile runner.

No CUDA package is imported at module import time.  ``run`` first validates
the source and option hashes, then imports CUDA Python at the exact operation
that requires it.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import os
import random
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from ...errors import ProfileRunError
from ...hardware.model import HardwareSpec
from ...model import TimingMetric
from ...profiles.model import ProfileEnvironment, profile_environment_id
from ..base import (
    CudaBenchmark,
    CudaBenchmarkCase,
    CudaBufferArgument,
    CudaBufferInit,
    CudaBufferRole,
    CudaScalarArgument,
    CudaScalarDType,
    MeasurementPolicy,
    NamedBufferOutput,
    ProfileRun,
    compile_options_sha256,
    cuda_source_sha256,
)


class LocalCudaProfileRunner:
    """Execute one exact benchmark on device zero of a local B200."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        if cache_dir is not None and not isinstance(cache_dir, Path):
            raise ProfileRunError("CUDA cache_dir must be Path or None")
        self.cache_dir = (
            cache_dir
            if cache_dir is not None
            else Path.home() / ".cache" / "tilefoundry-costmodel" / "nvrtc"
        )

    def run(
        self,
        benchmark: CudaBenchmark,
        *,
        hardware: HardwareSpec,
        policy: MeasurementPolicy,
    ) -> ProfileRun:
        if type(benchmark) is not CudaBenchmark:
            raise ProfileRunError("CUDA runner benchmark must be CudaBenchmark")
        if type(hardware) is not HardwareSpec:
            raise ProfileRunError("CUDA runner hardware must be HardwareSpec")
        if type(policy) is not MeasurementPolicy:
            raise ProfileRunError("CUDA runner policy must be MeasurementPolicy")
        if benchmark.key.query.hardware != hardware.ref:
            raise ProfileRunError("benchmark key hardware does not match runner hardware")

        # This check intentionally precedes optional imports.  A malformed or
        # substituted artifact is rejected even on a host without CUDA.
        fingerprint = benchmark.key.fingerprint
        if cuda_source_sha256(benchmark.source_utf8) != fingerprint.source_sha256:
            raise ProfileRunError("benchmark source hash does not match profile key")
        if compile_options_sha256(benchmark.compile_options) != fingerprint.compile_options_sha256:
            raise ProfileRunError("benchmark compile-option hash does not match profile key")

        try:
            driver = importlib.import_module("cuda.bindings.driver")
            nvrtc = importlib.import_module("cuda.bindings.nvrtc")
            runtime = importlib.import_module("cuda.bindings.runtime")
        except ImportError as exc:
            raise ProfileRunError("CUDA profiling requires the cuda optional dependency") from exc

        context: Any | None = None
        stream: Any | None = None
        module: Any | None = None
        try:
            _check_driver(driver.cuInit(0), "cuInit")
            device_count = int(_driver_value(driver.cuDeviceGetCount(), "cuDeviceGetCount"))
            if device_count <= 0:
                raise ProfileRunError("no CUDA device is visible")
            device = _driver_value(driver.cuDeviceGet(0), "cuDeviceGet")
            environment = _capture_environment(driver, nvrtc, runtime, device, hardware)
            context = _driver_value(driver.cuCtxCreate(0, device), "cuCtxCreate")
            stream = _driver_value(
                driver.cuStreamCreate(driver.CUstream_flags.CU_STREAM_NON_BLOCKING),
                "cuStreamCreate",
            )
            ptx = self._load_or_compile(nvrtc, benchmark, environment)
            module = _driver_value(driver.cuModuleLoadData(ptx), "cuModuleLoadData")

            latency_samples, latency_repetitions, latency_outputs = _run_case(
                driver,
                module,
                stream,
                benchmark.latency_case,
                policy,
            )
            interval_samples: tuple[int, ...] = ()
            interval_repetitions: int | None = None
            interval_outputs: tuple[NamedBufferOutput, ...] = ()
            if benchmark.initiation_interval_case is not None:
                interval_samples, interval_repetitions, interval_outputs = _run_case(
                    driver,
                    module,
                    stream,
                    benchmark.initiation_interval_case,
                    policy,
                )
            return ProfileRun(
                environment,
                latency_samples,
                interval_samples,
                latency_repetitions,
                interval_repetitions,
                latency_outputs + interval_outputs,
            )
        except ProfileRunError:
            raise
        except Exception as exc:
            raise ProfileRunError(f"CUDA benchmark execution failed: {exc}") from exc
        finally:
            if module is not None:
                _best_effort(driver, "cuModuleUnload", module)
            if stream is not None:
                _best_effort(driver, "cuStreamDestroy", stream)
            if context is not None:
                _best_effort(driver, "cuCtxDestroy", context)

    def _load_or_compile(
        self,
        nvrtc: Any,
        benchmark: CudaBenchmark,
        environment: ProfileEnvironment,
    ) -> bytes:
        cache_key = hashlib.sha256(
            (
                benchmark.key.fingerprint.source_sha256
                + benchmark.key.fingerprint.compile_options_sha256
                + environment.nvrtc_version
            ).encode("ascii")
        ).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.ptx"
        try:
            cached = cache_path.read_bytes()
            if cached:
                return cached
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ProfileRunError(f"cannot read NVRTC cache: {exc}") from exc

        program: Any | None = None
        try:
            result = nvrtc.nvrtcCreateProgram(
                benchmark.source_utf8.encode("utf-8"),
                b"tilefoundry_profile.cu",
                0,
                (),
                (),
            )
            program = _nvrtc_value(result, nvrtc, "nvrtcCreateProgram")
            options = tuple(option.encode("utf-8") for option in benchmark.compile_options)
            compile_result = nvrtc.nvrtcCompileProgram(program, len(options), options)
            if not _result_success(compile_result[0]):
                log = _nvrtc_log(nvrtc, program)
                raise ProfileRunError(f"NVRTC compilation failed: {log}")
            ptx_size = int(_nvrtc_value(nvrtc.nvrtcGetPTXSize(program), nvrtc, "nvrtcGetPTXSize"))
            ptx_buffer = bytearray(ptx_size)
            _check_nvrtc(nvrtc.nvrtcGetPTX(program, ptx_buffer), nvrtc, "nvrtcGetPTX")
            ptx = bytes(ptx_buffer)
            if not ptx:
                raise ProfileRunError("NVRTC produced an empty PTX artifact")
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=self.cache_dir, delete=False) as temporary:
                    temporary.write(ptx)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                    temporary_path = Path(temporary.name)
                os.replace(temporary_path, cache_path)
            except OSError as exc:
                raise ProfileRunError(f"cannot publish NVRTC cache artifact: {exc}") from exc
            return ptx
        finally:
            if program is not None:
                try:
                    nvrtc.nvrtcDestroyProgram(program)
                except Exception:
                    pass


def _capture_environment(
    driver: Any,
    nvrtc: Any,
    runtime: Any,
    device: Any,
    hardware: HardwareSpec,
) -> ProfileEnvironment:
    name_bytes = cast(bytes, _driver_value(driver.cuDeviceGetName(128, device), "cuDeviceGetName"))
    name = name_bytes.split(b"\0", 1)[0].decode("utf-8", errors="strict")
    major = int(
        _driver_value(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device,
            ),
            "cuDeviceGetAttribute(major)",
        )
    )
    minor = int(
        _driver_value(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device,
            ),
            "cuDeviceGetAttribute(minor)",
        )
    )
    if "B200" not in name or (major, minor) != (10, 0) or hardware.architecture != "B200":
        raise ProfileRunError(
            f"local CUDA device does not match B200 hardware: {name} sm_{major}{minor}"
        )
    uuid_value = _driver_value(driver.cuDeviceGetUuid_v2(device), "cuDeviceGetUuid_v2")
    uuid_bytes = bytes(getattr(uuid_value, "bytes"))
    driver_encoded = int(_driver_value(driver.cuDriverGetVersion(), "cuDriverGetVersion"))
    runtime_encoded = int(_runtime_value(runtime.cudaRuntimeGetVersion(), "cudaRuntimeGetVersion"))
    nvrtc_version_result = nvrtc.nvrtcVersion()
    _check_nvrtc(nvrtc_version_result, nvrtc, "nvrtcVersion")
    nvrtc_major = int(nvrtc_version_result[1])
    nvrtc_minor = int(nvrtc_version_result[2])
    device_clock = int(
        _driver_value(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                device,
            ),
            "cuDeviceGetAttribute(clock)",
        )
    )
    memory_clock = int(
        _driver_value(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                device,
            ),
            "cuDeviceGetAttribute(memory_clock)",
        )
    )
    provisional = ProfileEnvironment(
        "pending",
        uuid_bytes.hex(),
        hardware.ref,
        "sm_100a",
        _encoded_cuda_version(driver_encoded),
        _encoded_cuda_version(runtime_encoded),
        f"{nvrtc_major}.{nvrtc_minor}",
        device_clock if device_clock > 0 else None,
        memory_clock if memory_clock > 0 else None,
        None,
    )
    return replace(provisional, environment_id=profile_environment_id(provisional))


def _run_case(
    driver: Any,
    module: Any,
    stream: Any,
    case: CudaBenchmarkCase,
    policy: MeasurementPolicy,
) -> tuple[tuple[int, ...], int, tuple[NamedBufferOutput, ...]]:
    function = _driver_value(
        driver.cuModuleGetFunction(module, case.kernel_name.encode("ascii")),
        f"cuModuleGetFunction({case.kernel_name})",
    )
    buffers: dict[str, Any] = {}
    initial_data: dict[str, bytes] = {}
    try:
        for argument in case.arguments:
            if isinstance(argument, CudaBufferArgument):
                pointer = _driver_value(driver.cuMemAlloc(argument.nbytes), "cuMemAlloc")
                buffers[argument.name] = pointer
                data = _initial_buffer(argument)
                initial_data[argument.name] = data
                _check_driver(
                    driver.cuMemcpyHtoD(pointer, data, argument.nbytes),
                    "cuMemcpyHtoD",
                )

        # First use and requested warmups are synchronized outside retained
        # events.  They include module/function first-touch and cache effects.
        first_use = _case_with_repetitions(case, 1)
        _launch(driver, function, stream, first_use, buffers)
        _check_driver(driver.cuStreamSynchronize(stream), "cuStreamSynchronize(first-use)")
        for _ in range(policy.warmup_runs):
            _reset_buffers(driver, case, buffers, initial_data)
            _launch(driver, function, stream, first_use, buffers)
        if policy.warmup_runs:
            _check_driver(driver.cuStreamSynchronize(stream), "cuStreamSynchronize(warmup)")

        _reset_buffers(driver, case, buffers, initial_data)
        one_elapsed_ps = _time_launch(driver, function, stream, first_use, buffers)
        repetitions = max(
            1,
            min(
                policy.max_repetitions_per_sample,
                _ceil_div(policy.target_sample_ns * 1_000, one_elapsed_ps),
            ),
        )
        measured_case = _case_with_repetitions(case, repetitions)
        divisor = repetitions
        if case.metric is TimingMetric.INITIATION_INTERVAL:
            divisor *= _independent_chains(case)
        samples: list[int] = []
        for _ in range(policy.sample_count):
            _reset_buffers(driver, case, buffers, initial_data)
            elapsed_ps = _time_launch(driver, function, stream, measured_case, buffers)
            samples.append(max(1, elapsed_ps // divisor))

        # Correctness is evaluated on one semantic operation, independent of
        # the repetition count selected for timing.  This avoids treating
        # benchmark-chain scratch state as the public operation output.
        _reset_buffers(driver, case, buffers, initial_data)
        _launch(driver, function, stream, first_use, buffers)
        _check_driver(driver.cuStreamSynchronize(stream), "cuStreamSynchronize(correctness)")
        outputs: list[NamedBufferOutput] = []
        for argument in case.arguments:
            if isinstance(argument, CudaBufferArgument) and argument.role in (
                CudaBufferRole.OUTPUT,
                CudaBufferRole.INOUT,
            ):
                output = bytearray(argument.nbytes)
                _check_driver(
                    driver.cuMemcpyDtoH(output, buffers[argument.name], argument.nbytes),
                    "cuMemcpyDtoH",
                )
                outputs.append(NamedBufferOutput(case.metric, argument.name, bytes(output)))
        return tuple(samples), repetitions, tuple(outputs)
    finally:
        for pointer in buffers.values():
            _best_effort(driver, "cuMemFree", pointer)


def _time_launch(
    driver: Any,
    function: Any,
    stream: Any,
    case: CudaBenchmarkCase,
    buffers: dict[str, Any],
) -> int:
    start = _driver_value(
        driver.cuEventCreate(driver.CUevent_flags.CU_EVENT_DEFAULT), "cuEventCreate(start)"
    )
    end = _driver_value(
        driver.cuEventCreate(driver.CUevent_flags.CU_EVENT_DEFAULT), "cuEventCreate(end)"
    )
    try:
        prepared = _prepared_arguments(case, buffers)
        _check_driver(driver.cuEventRecord(start, stream), "cuEventRecord(start)")
        _launch_prepared(driver, function, stream, case, prepared)
        _check_driver(driver.cuEventRecord(end, stream), "cuEventRecord(end)")
        _check_driver(driver.cuEventSynchronize(end), "cuEventSynchronize(end)")
        elapsed_ms = float(
            _driver_value(driver.cuEventElapsedTime(start, end), "cuEventElapsedTime")
        )
        elapsed_ps = int(round(elapsed_ms * 1_000_000_000.0))
        if elapsed_ps <= 0:
            raise ProfileRunError("CUDA event interval was not positive")
        return elapsed_ps
    finally:
        _best_effort(driver, "cuEventDestroy", end)
        _best_effort(driver, "cuEventDestroy", start)


def _launch(
    driver: Any,
    function: Any,
    stream: Any,
    case: CudaBenchmarkCase,
    buffers: dict[str, Any],
) -> None:
    _launch_prepared(driver, function, stream, case, _prepared_arguments(case, buffers))


def _prepared_arguments(
    case: CudaBenchmarkCase, buffers: dict[str, Any]
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    parameter_values: list[Any] = []
    parameter_types: list[Any] = []
    for argument in case.arguments:
        if isinstance(argument, CudaBufferArgument):
            parameter_values.append(buffers[argument.name])
            parameter_types.append(None)
        else:
            parameter_values.append(argument.value)
            parameter_types.append(_scalar_ctype(argument))
    return (tuple(parameter_values), tuple(parameter_types))


def _launch_prepared(
    driver: Any,
    function: Any,
    stream: Any,
    case: CudaBenchmarkCase,
    prepared: tuple[tuple[Any, ...], tuple[Any, ...]],
) -> None:
    launch = case.launch
    result = driver.cuLaunchKernel(
        function,
        launch.grid[0],
        launch.grid[1],
        launch.grid[2],
        launch.block[0],
        launch.block[1],
        launch.block[2],
        launch.dynamic_smem_bytes,
        stream,
        prepared,
        0,
    )
    _check_driver(result, f"cuLaunchKernel({case.kernel_name})")


def _scalar_ctype(argument: CudaScalarArgument) -> Any:
    constructors: dict[CudaScalarDType, Any] = {
        CudaScalarDType.I32: ctypes.c_int32,
        CudaScalarDType.I64: ctypes.c_int64,
        CudaScalarDType.U32: ctypes.c_uint32,
        CudaScalarDType.U64: ctypes.c_uint64,
        CudaScalarDType.F32: ctypes.c_float,
        CudaScalarDType.F64: ctypes.c_double,
    }
    return constructors[argument.dtype]


def _case_with_repetitions(case: CudaBenchmarkCase, repetitions: int) -> CudaBenchmarkCase:
    arguments = tuple(
        replace(argument, value=repetitions)
        if isinstance(argument, CudaScalarArgument)
        and argument.name == case.repetition_argument_name
        else argument
        for argument in case.arguments
    )
    return replace(case, arguments=arguments)


def _independent_chains(case: CudaBenchmarkCase) -> int:
    for argument in case.arguments:
        if isinstance(argument, CudaScalarArgument) and argument.name == "independent_chains":
            if isinstance(argument.value, int) and argument.value > 0:
                return argument.value
    return 1


def _initial_buffer(argument: CudaBufferArgument) -> bytes:
    if argument.initialization is CudaBufferInit.ZERO:
        return bytes(argument.nbytes)
    if argument.initialization is CudaBufferInit.SEQUENCE:
        return bytes((argument.seed + index) % 251 for index in range(argument.nbytes))
    generator = random.Random(argument.seed)
    return bytes(generator.randrange(0, 256) for _ in range(argument.nbytes))


def _reset_buffers(
    driver: Any,
    case: CudaBenchmarkCase,
    buffers: dict[str, Any],
    initial_data: dict[str, bytes],
) -> None:
    for argument in case.arguments:
        if isinstance(argument, CudaBufferArgument):
            _check_driver(
                driver.cuMemcpyHtoD(
                    buffers[argument.name], initial_data[argument.name], argument.nbytes
                ),
                "cuMemcpyHtoD(reset)",
            )


def _encoded_cuda_version(value: int) -> str:
    major = value // 1000
    minor = (value % 1000) // 10
    return f"{major}.{minor}"


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _result_success(value: object) -> bool:
    try:
        return bool(int(cast(Any, value)) == 0)
    except (TypeError, ValueError):
        return str(value).endswith("SUCCESS")


def _check_driver(result: tuple[Any, ...], operation: str) -> None:
    if not isinstance(result, tuple) or not result or not _result_success(result[0]):
        raise ProfileRunError(f"{operation} failed: {result!r}")


def _driver_value(result: tuple[Any, ...], operation: str) -> Any:
    _check_driver(result, operation)
    if len(result) < 2:
        raise ProfileRunError(f"{operation} returned no value")
    return result[1]


def _check_nvrtc(result: tuple[Any, ...], nvrtc: Any, operation: str) -> None:
    del nvrtc
    if not isinstance(result, tuple) or not result or not _result_success(result[0]):
        raise ProfileRunError(f"{operation} failed: {result!r}")


def _nvrtc_value(result: tuple[Any, ...], nvrtc: Any, operation: str) -> Any:
    _check_nvrtc(result, nvrtc, operation)
    if len(result) < 2:
        raise ProfileRunError(f"{operation} returned no value")
    return result[1]


def _runtime_value(result: tuple[Any, ...], operation: str) -> Any:
    if not isinstance(result, tuple) or not result or not _result_success(result[0]):
        raise ProfileRunError(f"{operation} failed: {result!r}")
    if len(result) < 2:
        raise ProfileRunError(f"{operation} returned no value")
    return result[1]


def _nvrtc_log(nvrtc: Any, program: Any) -> str:
    try:
        size = int(
            _nvrtc_value(nvrtc.nvrtcGetProgramLogSize(program), nvrtc, "nvrtcGetProgramLogSize")
        )
        buffer = bytearray(size)
        _check_nvrtc(nvrtc.nvrtcGetProgramLog(program, buffer), nvrtc, "nvrtcGetProgramLog")
        return bytes(buffer).rstrip(b"\0").decode("utf-8", errors="replace")
    except ProfileRunError:
        return "<NVRTC log unavailable>"


def _best_effort(module: Any, name: str, *args: object) -> None:
    function = getattr(module, name, None)
    if callable(function):
        try:
            function(*args)
        except Exception:
            pass


__all__ = ["LocalCudaProfileRunner"]
