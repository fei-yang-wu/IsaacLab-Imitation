from __future__ import annotations

from pathlib import Path

import pytest

from imitation_experiments.evaluation.segment_semantic_phase_videos import (
    build_ffmpeg_command,
)


def test_build_ffmpeg_command_uses_end_exclusive_frame_trim(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "phase.mp4"

    command = build_ffmpeg_command(
        input_path=source, output_path=output, start_step=70, end_step=140
    )

    assert command[0] == "ffmpeg"
    assert "trim=start_frame=70:end_frame=140,setpts=PTS-STARTPTS" in command
    assert command[-1] == str(output)


def test_build_ffmpeg_command_rejects_empty_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid phase interval"):
        build_ffmpeg_command(
            input_path=tmp_path / "source.mp4",
            output_path=tmp_path / "phase.mp4",
            start_step=10,
            end_step=10,
        )
