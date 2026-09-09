from pathlib import Path
import math
import subprocess

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="FFmpeg GPU Service")


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = Path("/data").resolve()

FFMPEG = "/opt/ffmpeg/bin/ffmpeg"
FFPROBE = "/opt/ffmpeg/bin/ffprobe"


# ============================================================
# MODELS
# ============================================================

class EncodeRequest(BaseModel):
    input: str
    output: str

    video_codec: str = Field(
        default="h264_nvenc",
        pattern="^(h264_nvenc|hevc_nvenc|av1_nvenc)$",
    )

    preset: str = Field(
        default="p5",
        pattern="^p[1-7]$",
    )

    cq: int = Field(
        default=23,
        ge=0,
        le=51,
    )

    # Optional FFmpeg filter_complex.
    # Used by Add Visual.
    filter_complex: str | None = None

    audio_codec: str = "aac"
    audio_bitrate: str = "192k"

    overwrite: bool = True


class LoopMixRequest(BaseModel):
    video: str
    music: str
    output: str

    video_codec: str = Field(
        default="h264_nvenc",
        pattern="^(h264_nvenc|hevc_nvenc|av1_nvenc)$",
    )

    preset: str = Field(
        default="p5",
        pattern="^p[1-7]$",
    )

    cq: int = Field(
        default=23,
        ge=0,
        le=51,
    )

    music_volume: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
    )

    original_volume: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
    )

    transition_frames: int = Field(
        default=4,
        ge=0,
        le=24,
    )

    reverse: bool = False

    audio_codec: str = "aac"
    audio_bitrate: str = "192k"

    overwrite: bool = True


# ============================================================
# SAFE PATH
# ============================================================

def safe_path(value: str) -> Path:
    """
    Chỉ cho phép truy cập file bên trong /data.
    """

    path = (DATA_ROOT / value.lstrip("/")).resolve()

    try:
        path.relative_to(DATA_ROOT)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Path must remain inside /data",
        )

    return path


# ============================================================
# RUN COMMAND
# ============================================================

def run_command(
    command: list[str],
    timeout: int = 3600,
):
    print("=" * 80, flush=True)
    print("RUN COMMAND", flush=True)
    print(" ".join(command), flush=True)
    print("=" * 80, flush=True)

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print("FFMPEG TIMEOUT", flush=True)

        raise HTTPException(
            status_code=504,
            detail="FFmpeg process timed out",
        )

    print("=" * 80, flush=True)
    print(f"RETURN CODE: {result.returncode}", flush=True)
    print("STDOUT:", flush=True)
    print(result.stdout[-5000:], flush=True)
    print("STDERR:", flush=True)
    print(result.stderr[-20000:], flush=True)
    print("=" * 80, flush=True)

    return result


# ============================================================
# FFPROBE HELPERS
# ============================================================

def probe_value(
    args: list[str],
    label: str,
) -> str:

    result = run_command(
        [FFPROBE, "-v", "error", *args],
        timeout=60,
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Unable to probe {label}",
                "stderr": result.stderr[-5000:],
            },
        )

    value = result.stdout.strip()

    if not value:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to determine {label}",
        )

    return value


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            FFPROBE,
            "-v", "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "ffprobe failed",
                "path": str(path),
                "returncode": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
        )

    if stdout and stdout != "N/A":
        try:
            duration = float(stdout)

            if math.isfinite(duration) and duration > 0:
                return duration
        except ValueError:
            pass

    # --------------------------------------------------------
    # FALLBACK: let ffmpeg decode the stream to EOF
    # --------------------------------------------------------

    result = subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    stderr = result.stderr

    import re

    matches = re.findall(
        r"time=(\d+):(\d+):(\d+(?:\.\d+)?)",
        stderr,
    )

    if matches:
        hours, minutes, seconds = matches[-1]

        duration = (
            int(hours) * 3600
            + int(minutes) * 60
            + float(seconds)
        )

        if math.isfinite(duration) and duration > 0:
            return duration

    raise HTTPException(
        status_code=500,
        detail={
            "error": "Unable to determine media duration.",
            "path": str(path),
            "ffprobe_stdout": stdout,
            "ffprobe_stderr": stderr,
            "ffmpeg_stderr_tail": stderr[-5000:],
        },
    )


def probe_fps(path: Path) -> tuple[int, int]:
    value = probe_value(
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        "video FPS",
    )

    try:
        numerator, denominator = value.split("/")
        return int(numerator), int(denominator)

    except Exception:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid FPS: {value}",
        )


def probe_audio_sample_rate(path: Path) -> int:
    value = probe_value(
        [
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        "audio sample rate",
    )

    try:
        return int(value)
    except ValueError:
        return 48000

   
# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
    }


# ============================================================
# GPU
# ============================================================

@app.get("/gpu")
def gpu():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )

        return {
            "status": "ok",
            "gpu": result.stdout.strip(),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# CODECS
# ============================================================

@app.get("/codecs")
def codecs():
    result = subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-encoders",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    encoders = []

    for line in result.stdout.splitlines():
        if any(
            codec in line
            for codec in [
                "h264_nvenc",
                "hevc_nvenc",
                "av1_nvenc",
            ]
        ):
            encoders.append(line.strip())

    return {
        "status": "ok",
        "encoders": encoders,
    }


# ============================================================
# FFMPEG VERSION
# ============================================================

@app.get("/ffmpeg-version")
def ffmpeg_version():
    result = subprocess.run(
        [
            FFMPEG,
            "-version",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    return {
        "status": "ok",
        "version": (
            result.stdout.splitlines()[0]
            if result.stdout
            else ""
        ),
    }


# ============================================================
# SIMPLE ENCODE
# ============================================================

@app.post("/encode")
def encode(request: EncodeRequest):

    input_path = safe_path(request.input)
    output_path = safe_path(request.output)

    # ========================================================
    # VALIDATION
    # ========================================================

    if not input_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Input file not found: {request.input}",
        )

    if not input_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Input is not a file: {request.input}",
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # BUILD COMMAND
    # ========================================================

    command = [
        FFMPEG,
        "-hide_banner",
        "-loglevel", "error",

        "-y" if request.overwrite else "-n",

        "-i",
        str(input_path),
    ]

    # ========================================================
    # VIDEO FILTER / MAP
    # ========================================================

    if request.filter_complex:

        command.extend([
            "-filter_complex",
            request.filter_complex,

            "-map",
            "[v]",

            "-map",
            "0:a?",
        ])

    else:

        command.extend([
            "-map",
            "0:v:0",

            "-map",
            "0:a?",
        ])

    # ========================================================
    # VIDEO ENCODE
    # ========================================================

    command.extend([
        "-c:v",
        request.video_codec,

        "-preset",
        request.preset,

        "-cq",
        str(request.cq),

        "-pix_fmt",
        "yuv420p",
    ])

    # ========================================================
    # AUDIO
    # ========================================================

    command.extend([
        "-c:a",
        request.audio_codec,
    ])

    if request.audio_codec != "copy":

        command.extend([
            "-b:a",
            request.audio_bitrate,
        ])

    # ========================================================
    # OUTPUT
    # ========================================================

    command.append(
        str(output_path)
    )

    # ========================================================
    # EXECUTE
    # ========================================================

    result = run_command(command)

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "returncode": result.returncode,
                "stderr": result.stderr[-10000:],
            },
        )

    # ========================================================
    # OUTPUT VALIDATION
    # ========================================================

    if not output_path.exists():
        raise HTTPException(
            status_code=500,
            detail="FFmpeg completed but output file was not created.",
        )

    if output_path.stat().st_size == 0:
        raise HTTPException(
            status_code=500,
            detail="FFmpeg created an empty output file.",
        )

    # ========================================================
    # RESPONSE
    # ========================================================

    duration_seconds = probe_duration(output_path)

    return {
        "status": "ok",
        "input": request.input,
        "output": request.output,
        "video_codec": request.video_codec,
        "preset": request.preset,
        "cq": request.cq,
        "audio_codec": request.audio_codec,
        "filter_complex": bool(request.filter_complex),
        "size_bytes": output_path.stat().st_size,
        "duration_seconds": duration_seconds,
    }

# ============================================================
# LOOP + REVERSE + CURVE DISSOLVE + AUDIO MIX
# ============================================================

def probe_duration_with_ffmpeg(path: Path) -> float:
    """
    Determine media duration using FFmpeg itself.

    Used when ffprobe cannot determine duration,
    especially WebM/Opus files without duration metadata.
    """

    result = subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel", "info",
            "-i",
            str(path),
            "-map", "0:a:0",
            "-f", "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )

    stderr = result.stderr or ""

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "FFmpeg could not determine audio duration",
                "path": str(path),
                "returncode": result.returncode,
                "stderr": stderr[-10000:],
            },
        )

    # --------------------------------------------------------
    # Parse final FFmpeg time= value.
    #
    # Example:
    # time=00:06:00.12
    # --------------------------------------------------------

    duration = None

    for line in stderr.splitlines():

        marker = "time="

        if marker not in line:
            continue

        value = line.rsplit(marker, 1)[-1].strip()

        value = value.split()[0]

        if value == "N/A":
            continue

        parts = value.split(":")

        if len(parts) != 3:
            continue

        try:

            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])

            candidate = (
                hours * 3600.0
                + minutes * 60.0
                + seconds
            )

            if math.isfinite(candidate) and candidate > 0:
                duration = candidate

        except ValueError:
            continue

    if duration is None:
        raise HTTPException(
            status_code=500,
            detail={
                "error": (
                    "Unable to determine media duration "
                    "with FFmpeg"
                ),
                "path": str(path),
                "ffmpeg_stderr": stderr[-20000:],
            },
        )

    return duration


def probe_duration_resilient(path: Path) -> float:
    """
    First try ffprobe.

    If duration is unavailable, let FFmpeg decode
    the media to EOF and determine duration.
    """

    try:
        return probe_duration(path)

    except HTTPException:
        return probe_duration_with_ffmpeg(path)


@app.post("/loop-mix")
def loop_mix(request: LoopMixRequest):

    # ========================================================
    # PATHS
    # ========================================================

    video_path = safe_path(request.video)
    music_path = safe_path(request.music)
    output_path = safe_path(request.output)

    # ========================================================
    # VALIDATION
    # ========================================================

    if not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Video file not found: "
                f"{request.video}"
            ),
        )

    if not music_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Music file not found: "
                f"{request.music}"
            ),
        )

    if not video_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Video is not a file: "
                f"{request.video}"
            ),
        )

    if not music_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Music is not a file: "
                f"{request.music}"
            ),
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # PROBE MEDIA
    # ========================================================

    video_duration = probe_duration(video_path)

    # Suno WebM/Opus may return N/A from ffprobe.
    # FFmpeg itself becomes the fallback source of truth.
    music_duration = probe_duration_resilient(
        music_path
    )

    fps_num, fps_den = probe_fps(video_path)

    fps = fps_num / fps_den

    if fps <= 0:
        raise HTTPException(
            status_code=500,
            detail="Invalid video FPS",
        )

    # ========================================================
    # TRANSITION
    # ========================================================

    transition_frames = request.transition_frames

    if transition_frames > 0:

        transition_duration = (
            transition_frames / fps
        )

    else:

        transition_duration = 0.0

    # Never allow transition to consume
    # too much of the source clip.

    max_transition = video_duration * 0.25

    if transition_duration > max_transition:

        transition_duration = max_transition

    # ========================================================
    # SEGMENT COUNT
    #
    # With xfade:
    #
    # total duration =
    #
    # N * video_duration
    # -
    # (N - 1) * transition_duration
    #
    # ========================================================

    if transition_duration > 0:

        effective_segment_duration = (
            video_duration
            - transition_duration
        )

        if effective_segment_duration <= 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Transition duration is too long "
                    "for the source video."
                ),
            )

        segment_count = max(
            2,
            math.ceil(
                (
                    music_duration
                    - transition_duration
                )
                / effective_segment_duration
            ) + 1,
        )

    else:

        segment_count = max(
            1,
            math.ceil(
                music_duration
                / video_duration
            ),
        )

    # ========================================================
    # BUILD FFMPEG INPUTS
    #
    # One input per video segment.
    #
    # This lets each segment independently become:
    #
    # NORMAL
    # NORMAL
    # REVERSE
    # REVERSE
    #
    # Pattern:
    #
    # 0 NORMAL
    # 1 NORMAL
    # 2 REVERSE
    # 3 REVERSE
    # 4 NORMAL
    # 5 NORMAL
    # 6 REVERSE
    # 7 REVERSE
    # ...
    # ========================================================

    command = [
        FFMPEG,
        "-hide_banner",
        "-loglevel", "error",
        "-y" if request.overwrite else "-n",
    ]

    for _ in range(segment_count):

        command.extend(
            [
                "-i",
                str(video_path),
            ]
        )

    # Music is the final input.

    music_input_index = segment_count

    command.extend(
        [
            "-i",
            str(music_path),
        ]
    )

    # ========================================================
    # FILTER GRAPH
    # ========================================================

    filters: list[str] = []

    # ========================================================
    # VIDEO SEGMENTS
    #
    # reverse=True:
    #
    # index % 4:
    #
    # 0 = NORMAL
    # 1 = NORMAL
    # 2 = REVERSE
    # 3 = REVERSE
    #
    # Therefore:
    #
    # NORMAL
    # NORMAL
    # REVERSE
    # REVERSE
    # NORMAL
    # NORMAL
    # REVERSE
    # REVERSE
    #
    # This is the Premiere-style pattern.
    # ========================================================

    for index in range(segment_count):

        input_label = f"{index}:v"

        pattern_position = index % 4

        should_reverse = (
            request.reverse
            and pattern_position in (2, 3)
        )

        if should_reverse:

            filters.append(
                (
                    f"[{input_label}]"
                    "setpts=PTS-STARTPTS,"
                    "reverse,"
                    "setpts=PTS-STARTPTS"
                    f"[v{index}]"
                )
            )

        else:

            filters.append(
                (
                    f"[{input_label}]"
                    "setpts=PTS-STARTPTS"
                    f"[v{index}]"
                )
            )

    # ========================================================
    # VIDEO TRANSITION
    #
    # Premiere-like:
    #
    # Film Dissolve + Curve
    #
    # Instead of:
    #
    # transition=dissolve
    #
    # use custom xfade expression.
    #
    # P = transition progress 0..1
    #
    # Curve:
    #
    # P*P*(3-2*P)
    #
    # This is smoothstep:
    #
    # 0%   -> 0%
    # 25%  -> 15.625%
    # 50%  -> 50%
    # 75%  -> 84.375%
    # 100% -> 100%
    #
    # This gives softer entry/exit and more natural ghosting.
    # ========================================================

    if segment_count == 1:

        filters.append(
            (
                "[v0]"
                f"trim=duration={music_duration:.6f},"
                "setpts=PTS-STARTPTS"
                "[vout]"
            )
        )

    else:

        current = "v0"

        accumulated_duration = video_duration

        for index in range(1, segment_count):

            next_label = f"v{index}"

            output_label = f"xf{index}"

            offset = (
                accumulated_duration
                - transition_duration
            )

            if transition_duration > 0:

                # ------------------------------------------------
                # CUSTOM CURVE
                #
                # A = first input pixel
                # B = second input pixel
                # P = transition progress
                #
                # smoothstep(P):
                #
                # P*P*(3-2*P)
                #
                # Output:
                #
                # A*(1-CURVE) + B*CURVE
                #
                # ------------------------------------------------

                curve_expr = (
                    "A*(1-(P*P*(3-2*P)))"
                    "+B*(P*P*(3-2*P))"
                )

                filters.append(
                    (
                        f"[{current}]"
                        f"[{next_label}]"
                        "xfade="
                        "transition=custom:"
                        f"duration={transition_duration:.6f}:"
                        f"offset={offset:.6f}:"
                        f"expr='{curve_expr}'"
                        f"[{output_label}]"
                    )
                )

            else:

                filters.append(
                    (
                        f"[{current}]"
                        f"[{next_label}]"
                        "concat=n=2:v=1:a=0,"
                        "setpts=PTS-STARTPTS"
                        f"[{output_label}]"
                    )
                )

            current = output_label

            accumulated_duration = (
                accumulated_duration
                + video_duration
                - transition_duration
            )

        # ========================================================
        # FINAL VIDEO DURATION
        #
        # Music is the master duration.
        # ========================================================

        filters.append(
            (
                f"[{current}]"
                f"trim=duration={music_duration:.6f},"
                "setpts=PTS-STARTPTS"
                "[vout]"
            )
        )

    # ========================================================
    # AUDIO
    #
    # ORIGINAL VIDEO AUDIO = 100%
    # SUNO MUSIC            = 50%
    #
    # Original audio loops with the video.
    # Music determines final duration.
    # ========================================================

    original_audio_samples = max(
        1,
        int(
            math.ceil(
                video_duration * 48000
            )
        ),
    )

    filters.append(
        (
            "[0:a]"
            "aresample=48000,"
            "asetpts=PTS-STARTPTS,"
            f"aloop="
            f"loop=-1:"
            f"size={original_audio_samples},"
            f"atrim=duration={music_duration:.6f},"
            "asetpts=PTS-STARTPTS,"
            f"volume={request.original_volume:.6f}"
            "[original]"
        )
    )

    filters.append(
        (
            f"[{music_input_index}:a]"
            "aresample=48000,"
            "asetpts=PTS-STARTPTS,"
            f"atrim=duration={music_duration:.6f},"
            f"volume={request.music_volume:.6f}"
            "[music]"
        )
    )

    filters.append(
        (
            "[music][original]"
            "amix="
            "inputs=2:"
            "duration=first:"
            "dropout_transition=2:"
            "normalize=1"
            "[aout]"
        )
    )

    # ========================================================
    # FILTER COMPLEX
    # ========================================================

    filter_complex = ";".join(filters)

    # ========================================================
    # ENCODE
    # ========================================================

    command.extend(
        [
            "-filter_complex",
            filter_complex,

            "-map",
            "[vout]",

            "-map",
            "[aout]",

            "-c:v",
            request.video_codec,

            "-preset",
            request.preset,

            "-cq",
            str(request.cq),

            "-pix_fmt",
            "yuv420p",

            "-c:a",
            request.audio_codec,

            "-b:a",
            request.audio_bitrate,

            "-ar",
            "48000",

            "-t",
            f"{music_duration:.6f}",

            "-movflags",
            "+faststart",

            str(output_path),
        ]
    )

    # ========================================================
    # LOG
    # ========================================================

    print("=" * 70)
    print("LOOP MIX")
    print("=" * 70)

    print(
        f"VIDEO:                 "
        f"{video_path}"
    )

    print(
        f"MUSIC:                 "
        f"{music_path}"
    )

    print(
        f"OUTPUT:                "
        f"{output_path}"
    )

    print(
        f"VIDEO DURATION:        "
        f"{video_duration:.6f}s"
    )

    print(
        f"MUSIC DURATION:        "
        f"{music_duration:.6f}s"
    )

    print(
        f"FPS:                   "
        f"{fps_num}/{fps_den}"
    )

    print(
        f"TRANSITION FRAMES:     "
        f"{transition_frames}"
    )

    print(
        f"TRANSITION DURATION:   "
        f"{transition_duration:.6f}s"
    )

    print(
        f"SEGMENTS:              "
        f"{segment_count}"
    )

    print(
        f"REVERSE PATTERN:       "
        f"{request.reverse}"
    )

    if request.reverse:

        print(
            "PATTERN:               "
            "NORMAL -> NORMAL -> "
            "REVERSE -> REVERSE"
        )

    else:

        print(
            "PATTERN:               "
            "NORMAL -> NORMAL -> "
            "NORMAL -> NORMAL"
        )

    print(
        "TRANSITION CURVE:      "
        "smoothstep"
    )

    print(
        "CURVE FORMULA:         "
        "P*P*(3-2*P)"
    )

    print(
        f"ORIGINAL VOLUME:       "
        f"{request.original_volume:.3f}"
    )

    print(
        f"MUSIC VOLUME:          "
        f"{request.music_volume:.3f}"
    )

    print("=" * 70)

    # ========================================================
    # EXECUTE
    # ========================================================

    result = run_command(
        command,
        timeout=3600,
    )

    if result.returncode != 0:

        raise HTTPException(
            status_code=500,
            detail={
                "returncode": result.returncode,
                "stderr": result.stderr[-20000:],
            },
        )

    # ========================================================
    # VALIDATE OUTPUT
    # ========================================================

    if not output_path.exists():

        raise HTTPException(
            status_code=500,
            detail=(
                "FFmpeg completed but "
                "output file was not created"
            ),
        )

    if output_path.stat().st_size == 0:

        raise HTTPException(
            status_code=500,
            detail=(
                "FFmpeg created an empty output file"
            ),
        )

    # ========================================================
    # OUTPUT DURATION
    # ========================================================

    output_duration = probe_duration(
        output_path
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "status": "ok",

        "input": request.video,

        "music": request.music,

        "output": request.output,

        "video_codec": request.video_codec,

        "preset": request.preset,

        "cq": request.cq,

        "fps": {
            "numerator": fps_num,
            "denominator": fps_den,
            "value": fps,
        },

        "video_duration_seconds": (
            video_duration
        ),

        "music_duration_seconds": (
            music_duration
        ),

        "output_duration_seconds": (
            output_duration
        ),

        "segment_count": (
            segment_count
        ),

        "transition_frames": (
            transition_frames
        ),

        "transition_duration_seconds": (
            transition_duration
        ),

        "reverse_pattern": (
            "NORMAL_NORMAL_REVERSE_REVERSE"
            if request.reverse
            else "NORMAL_NORMAL_NORMAL_NORMAL"
        ),

        "transition": {
            "type": "custom_curve_dissolve",
            "curve": "smoothstep",
            "formula": "P*P*(3-2*P)",
        },

        "original_volume": (
            request.original_volume
        ),

        "music_volume": (
            request.music_volume
        ),

        "audio_codec": (
            request.audio_codec
        ),

        "audio_bitrate": (
            request.audio_bitrate
        ),

        "size_bytes": (
            output_path.stat().st_size
        ),
    }
