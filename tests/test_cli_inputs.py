import sys

import pytest
from PIL import Image

from wallgraph.cli import main


@pytest.mark.parametrize("name", ("walls.png", "walls.json", "walls.svg"))
@pytest.mark.parametrize("alias", ["path", "hardlink", "symlink"])
def test_detect_preserves_original_input_before_any_output(
    tmp_path, monkeypatch, capsys, name, alias
):
    output = tmp_path / "result"
    output.mkdir()
    source = output / name if alias == "path" else tmp_path / "source.png"
    Image.new("L", (32, 32), 255).save(source, format="PNG")
    target = output / name
    if alias == "hardlink":
        target.hardlink_to(source)
    elif alias == "symlink":
        target.symlink_to(source)
    original = source.read_bytes()
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    monkeypatch.setattr(
        sys, "argv", ["wallgraph", "detect", "--image", str(source), "--output", str(output)]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert source.read_bytes() == original
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    assert "choose a different --output directory" in capsys.readouterr().err


@pytest.mark.parametrize("problem", ["missing", "invalid_image", "output_is_file"])
def test_common_input_errors_exit_two_without_a_traceback(tmp_path, monkeypatch, capsys, problem):
    source = tmp_path / "input.png"
    output = tmp_path / "result"
    if problem == "invalid_image":
        source.write_text("This is not an image")
    if problem == "output_is_file":
        output.write_text("keep this file")
        args = ["wallgraph", "demo", "--output", str(output)]
    else:
        args = ["wallgraph", "detect", "--image", str(source), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "wallgraph:" in stderr and "Traceback" not in stderr
    if problem == "output_is_file":
        assert output.read_text() == "keep this file"
    else:
        assert not output.exists()
