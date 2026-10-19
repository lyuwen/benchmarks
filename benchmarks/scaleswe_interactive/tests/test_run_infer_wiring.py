import json

from benchmarks.scaleswe_interactive.run_infer import (
    build_arg_parser,
    load_llm_from_config,
)


def test_parser_has_interactive_args():
    parser = build_arg_parser()
    ns = parser.parse_args(["x.json"])
    assert ns.mode == "plan"                 # default
    assert ns.user_tools == "none"           # default
    assert ns.max_user_turns == 20           # default
    assert hasattr(ns, "user_llm_config_path")


def test_parser_accepts_overrides():
    parser = build_arg_parser()
    ns = parser.parse_args([
        "x.json",
        "--user-llm-config-path", "y.json",
        "--mode", "auto",
        "--user-tools", "readonly",
        "--max-user-turns", "5",
    ])
    assert ns.mode == "auto"
    assert ns.user_tools == "readonly"
    assert ns.max_user_turns == 5
    assert ns.user_llm_config_path == "y.json"


def test_load_llm_from_config(tmp_path):
    cfg = tmp_path / "llm.json"
    cfg.write_text(json.dumps({"model": "litellm_proxy/x", "api_key": "sk-test"}))
    llm = load_llm_from_config(str(cfg))
    assert llm.model == "litellm_proxy/x"
