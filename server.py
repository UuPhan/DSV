from pathlib import Path
import math
import subprocess
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Version 1
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

    reverse: bool = False

    audio_codec: str = "aac"
    audio_bitrate: str = "192k"

    overwrite: bool = True


class BurnLyricsRequest(BaseModel):
    video: str
    lyrics: str
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

    audio_codec: str = "aac"
    audio_bitrate: str = "192k"

    overwrite: bool = True

class PrependIntroRequest(BaseModel):
    intro: str
    main: str
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

    audio_codec: str = "aac"
    audio_bitrate: str = "192k"

    overwrite: bool = True
    
# ============================================================
# SAFE PATH
# ============================================================

def safe_path(value: str) -> Path:
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
# FILE VALIDATION
# ============================================================

def require_file(path: Path, label: str):
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{label} not found: {path}",
        )

    if not path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"{label} is not a file: {path}",
        )


def prepare_output(path: Path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def validate_output(path: Path):
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail="FFmpeg completed but output file was not created.",
        )

    if path.stat().st_size == 0:
        raise HTTPException(
            status_code=500,
            detail="FFmpeg created an empty output file.",
        )


# ============================================================
# COMMAND
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
        raise HTTPException(
            status_code=504,
            detail="FFmpeg process timed out.",
        )

    print(f"RETURN CODE: {result.returncode}", flush=True)

    if result.stdout:
        print(
            "STDOUT:",
            result.stdout[-5000:],
            flush=True,
        )

    if result.stderr:
        print(
            "STDERR:",
            result.stderr[-20000:],
            flush=True,
        )

    return result


def run_ffmpeg(
    command: list[str],
    timeout: int = 3600,
):
    result = run_command(
        command,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "returncode": result.returncode,
                "stderr": result.stderr[-20000:],
            },
        )

    return result


# ============================================================
# FFPROBE
# ============================================================

def probe_value(
    args: list[str],
    label: str,
) -> str:
    result = run_command(
        [
            FFPROBE,
            "-v",
            "error",
            *args,
        ],
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
            "-v",
            "error",
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

    if result.returncode == 0 and stdout and stdout != "N/A":
        try:
            duration = float(stdout)

            if math.isfinite(duration) and duration > 0:
                return duration

        except ValueError:
            pass

    return probe_duration_with_ffmpeg(path)


def probe_duration_with_ffmpeg(path: Path) -> float:
    result = subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(path),
            "-map",
            "0:a:0?",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )

    stderr = result.stderr or ""

    duration = None

    for match in re.finditer(
        r"time=(\d+):(\d+):(\d+(?:\.\d+)?)",
        stderr,
    ):
        hours = float(match.group(1))
        minutes = float(match.group(2))
        seconds = float(match.group(3))

        candidate = (
            hours * 3600.0
            + minutes * 60.0
            + seconds
        )

        if math.isfinite(candidate) and candidate > 0:
            duration = candidate

    if duration is not None:
        return duration

    raise HTTPException(
        status_code=500,
        detail={
            "error": "Unable to determine media duration.",
            "path": str(path),
            "ffmpeg_stderr": stderr[-10000:],
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

        numerator = int(numerator)
        denominator = int(denominator)

        if numerator <= 0 or denominator <= 0:
            raise ValueError

        return numerator, denominator

    except Exception:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid FPS: {value}",
        )


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
            for codec in (
                "h264_nvenc",
                "hevc_nvenc",
                "av1_nvenc",
            )
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

    require_file(
        input_path,
        "Input file",
    )

    prepare_output(output_path)

    command = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if request.overwrite else "-n",

        "-i",
        str(input_path),
    ]

    # --------------------------------------------------------
    # VIDEO
    #
    # Có filter_complex:
    #   [v] = video đã được filter, ví dụ drawtext.
    #
    # Không có filter:
    #   dùng video gốc.
    # --------------------------------------------------------

    if request.filter_complex:
        command.extend([
            "-filter_complex",
            request.filter_complex,

            "-map",
            "[v]",
        ])
    else:
        command.extend([
            "-map",
            "0:v:0",
        ])

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    command.extend([
        "-map",
        "0:a?",
    ])

    # --------------------------------------------------------
    # VIDEO ENCODE
    #
    # Vì video được encode lại nên drawtext đã trở thành pixel.
    # Không có subtitle stream được tạo ra.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # AUDIO ENCODE
    # --------------------------------------------------------

    command.extend([
        "-c:a",
        request.audio_codec,
    ])

    if request.audio_codec != "copy":
        command.extend([
            "-b:a",
            request.audio_bitrate,
        ])

    command.append(
        str(output_path)
    )

    run_ffmpeg(command)

    validate_output(output_path)

    duration_seconds = probe_duration(
        output_path
    )

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
# LOOP + MIX AUDIO
#
# NHIỆM VỤ DUY NHẤT:
#
# 1. Lặp video cho đủ thời lượng music.
# 2. Reverse video nếu được yêu cầu.
# 3. Mix audio gốc + music.
#
# KHÔNG:
# - xfade
# - transition
# - blend
# - smoothstep
# - transition frames
# - visual effect khác
# ============================================================

@app.post("/loop-mix")
def loop_mix(request: LoopMixRequest):
    video_path = safe_path(request.video)
    music_path = safe_path(request.music)
    output_path = safe_path(request.output)

    require_file(video_path, "Video file")
    require_file(music_path, "Music file")

    prepare_output(output_path)

    video_duration = probe_duration(video_path)
    music_duration = probe_duration(music_path)

    if video_duration <= 0:
        raise HTTPException(
            status_code=400,
            detail="Video duration must be greater than zero.",
        )

    if music_duration <= 0:
        raise HTTPException(
            status_code=400,
            detail="Music duration must be greater than zero.",
        )

    # Số segment cần để phủ hết thời lượng music.
    segment_count = max(
        1,
        math.ceil(music_duration / video_duration),
    )

    command = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if request.overwrite else "-n",

        # Chỉ một video input.
        "-i",
        str(video_path),

        # Music input.
        "-i",
        str(music_path),
    ]

    filters = []

    # --------------------------------------------------------
    # VIDEO
    #
    # reverse=false:
    #   normal → normal → normal → ...
    #
    # reverse=true:
    #   normal → normal → reverse → reverse → ...
    #   normal → normal → reverse → reverse → ...
    # --------------------------------------------------------

    filters.append(
        "[0:v]"
        "setpts=PTS-STARTPTS"
        "[v_normal]"
    )

    if request.reverse:
        filters.append(
            "[0:v]"
            "setpts=PTS-STARTPTS,"
            "reverse,"
            "setpts=PTS-STARTPTS"
            "[v_reverse]"
        )

        video_sequence = []

        for index in range(segment_count):
            pattern_index = index % 4

            if pattern_index in (0, 1):
                video_sequence.append("[v_normal]")
            else:
                video_sequence.append("[v_reverse]")

        filters.append(
            "".join(video_sequence)
            + f"concat=n={segment_count}:v=1:a=0,"
            + f"trim=duration={music_duration:.6f},"
            + "setpts=PTS-STARTPTS"
            + "[vout]"
        )

    else:
        filters.append(
            "".join(
                "[v_normal]"
                for _ in range(segment_count)
            )
            + f"concat=n={segment_count}:v=1:a=0,"
            + f"trim=duration={music_duration:.6f},"
            + "setpts=PTS-STARTPTS"
            + "[vout]"
        )

    # --------------------------------------------------------
    # ORIGINAL AUDIO
    #
    # Audio gốc của visual được loop/pad tới duration music.
    # Nếu visual không có audio thì bỏ qua.
    # --------------------------------------------------------

    filters.append(
        "[0:a]"
        "aresample=48000,"
        "asetpts=PTS-STARTPTS,"
        "apad,"
        f"atrim=duration={music_duration:.6f},"
        "asetpts=PTS-STARTPTS,"
        f"volume={request.original_volume:.6f}"
        "[original]"
    )

    # --------------------------------------------------------
    # MUSIC
    # --------------------------------------------------------

    filters.append(
        "[1:a]"
        "aresample=48000,"
        "asetpts=PTS-STARTPTS,"
        f"atrim=duration={music_duration:.6f},"
        "asetpts=PTS-STARTPTS,"
        f"volume={request.music_volume:.6f}"
        "[music]"
    )

    # --------------------------------------------------------
    # MIX AUDIO
    # --------------------------------------------------------

    filters.append(
        "[original][music]"
        "amix="
        "inputs=2:"
        "duration=longest:"
        "dropout_transition=0:"
        "normalize=0"
        "[aout]"
    )

    filter_complex = ";".join(filters)

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

    run_ffmpeg(
        command,
        timeout=3600,
    )

    validate_output(output_path)

    output_duration = probe_duration(output_path)

    return {
        "status": "ok",
        "input": request.video,
        "music": request.music,
        "output": request.output,

        "video_duration_seconds": video_duration,
        "music_duration_seconds": music_duration,
        "output_duration_seconds": output_duration,

        "segment_count": segment_count,

        "reverse": request.reverse,

        "pattern": (
            "NORMAL_NORMAL_REVERSE_REVERSE"
            if request.reverse
            else "NORMAL"
        ),

        "original_volume": request.original_volume,
        "music_volume": request.music_volume,

        "video_codec": request.video_codec,
        "preset": request.preset,
        "cq": request.cq,

        "size_bytes": output_path.stat().st_size,
    }


# ============================================================
# BURN LYRICS
#
# Hỗ trợ:
# - .srt
# - .ass
#
# Lyrics được burn trực tiếp vào video.
# Audio được giữ nguyên.
# ============================================================

@app.post("/burn-lyrics")
def burn_lyrics(request: BurnLyricsRequest):
    video_path = safe_path(request.video)
    lyrics_path = safe_path(request.lyrics)
    output_path = safe_path(request.output)

    require_file(
        video_path,
        "Video file",
    )

    require_file(
        lyrics_path,
        "Lyrics file",
    )

    suffix = lyrics_path.suffix.lower()

    if suffix not in (".srt", ".ass"):
        raise HTTPException(
            status_code=400,
            detail="Lyrics file must be .srt or .ass",
        )

    prepare_output(output_path)

    # FFmpeg subtitles filter.
    #
    # Escape characters required by the filter parser.
    subtitle_path = str(
        lyrics_path
    ).replace(
        "\\",
        "\\\\",
    ).replace(
        ":",
        "\\:",
    ).replace(
        "'",
        "\\'",
    )

    subtitle_filter = (
        f"subtitles='{subtitle_path}'"
    )

    command = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if request.overwrite else "-n",

        "-i",
        str(video_path),

        "-vf",
        subtitle_filter,

        "-map",
        "0:v:0",

        "-map",
        "0:a?",

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
    ]

    if request.audio_codec != "copy":
        command.extend(
            [
                "-b:a",
                request.audio_bitrate,
            ]
        )

    command.extend(
        [
            "-movflags",
            "+faststart",

            str(output_path),
        ]
    )

    run_ffmpeg(
        command,
        timeout=3600,
    )

    validate_output(output_path)
    
    duration_seconds = probe_duration(output_path)

    return {
        "status": "ok",
        "video": request.video,
        "lyrics": request.lyrics,
        "output": request.output,
        "lyrics_format": suffix.lstrip("."),
        "video_codec": request.video_codec,
        "preset": request.preset,
        "cq": request.cq,
        "size_bytes": output_path.stat().st_size,
        "duration_seconds": duration_seconds
    }
    
@app.post("/prepend-intro")
def prepend_intro(request: PrependIntroRequest):
    intro_path = safe_path(request.intro)
    main_path = safe_path(request.main)
    output_path = safe_path(request.output)

    require_file(intro_path, "Intro video")
    require_file(main_path, "Main video")
    prepare_output(output_path)

    command = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if request.overwrite else "-n",

        "-i",
        str(intro_path),

        "-i",
        str(main_path),

        "-filter_complex",
        (
            "[0:v:0]setpts=PTS-STARTPTS[intro_v];"
            "[1:v:0]setpts=PTS-STARTPTS[main_v];"
            "[0:a:0]aresample=48000,"
            "asetpts=PTS-STARTPTS[intro_a];"
            "[1:a:0]aresample=48000,"
            "asetpts=PTS-STARTPTS[main_a];"
            "[intro_v][intro_a][main_v][main_a]"
            "concat=n=2:v=1:a=1"
            "[vout][aout]"
        ),

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

        "-movflags",
        "+faststart",

        str(output_path),
    ]

    run_ffmpeg(command)
    validate_output(output_path)

    duration_seconds = probe_duration(output_path)

    return {
        "status": "ok",
        "intro": request.intro,
        "main": request.main,
        "output": request.output,
        "video_codec": request.video_codec,
        "preset": request.preset,
        "cq": request.cq,
        "audio_codec": request.audio_codec,
        "duration_seconds": duration_seconds,
        "size_bytes": output_path.stat().st_size,
    }    
