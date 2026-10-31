import io
import json

from nemotron_transcribe_mcp.logsetup import configure_logging, get_logger


def test_info_line_is_one_json_object() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    get_logger("server").info("server_started", host="127.0.0.1")

    line = json.loads(stream.getvalue())
    assert line["level"] == "INFO"
    assert line["logger"] == "nemotron.server"
    assert line["event"] == "server_started"
    assert line["host"] == "127.0.0.1"
    assert line["ts"].endswith("Z")


def test_debug_is_hidden_until_the_level_is_lowered() -> None:
    quiet = io.StringIO()
    configure_logging("INFO", stream=quiet)
    get_logger("server").debug("hidden_detail", audio_s=1)
    assert quiet.getvalue() == ""

    verbose = io.StringIO()
    configure_logging("DEBUG", stream=verbose)
    get_logger("server").debug("hidden_detail", audio_s=1)
    line = json.loads(verbose.getvalue())
    assert line["event"] == "hidden_detail"
    assert line["audio_s"] == 1


def test_text_format_is_one_readable_line() -> None:
    stream = io.StringIO()
    configure_logging("INFO", fmt="text", stream=stream)
    get_logger("server").info("server_started", host="127.0.0.1")

    line = stream.getvalue().strip()
    assert line.endswith("INFO nemotron.server server_started host=127.0.0.1")
    assert not line.startswith("{")


def test_api_key_is_redacted() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream, secrets=["super-secret"])
    get_logger("server").warning("header_rejected", detail="Bearer super-secret")

    text = stream.getvalue()
    assert "super-secret" not in text
    assert "<redacted>" in text
