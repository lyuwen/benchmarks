import os
from jinja2 import Environment, FileSystemLoader

PROMPTS = os.path.join(os.path.dirname(__file__), "..", "prompts")


def _render(name, **ctx):
    env = Environment(loader=FileSystemLoader(PROMPTS))
    return env.get_template(name).render(**ctx)


def test_user_system_plan_mentions_approval_and_finish():
    txt = _render("user_system.j2", problem_statement="P", mode="plan",
                  user_turns=0, max_user_turns=20)
    assert "plan" in txt.lower()
    assert "approve" in txt.lower()
    assert "finish" in txt.lower()
    assert "read-only" in txt.lower()


def test_user_system_auto_mentions_autonomy():
    txt = _render("user_system.j2", problem_statement="P", mode="auto",
                  user_turns=0, max_user_turns=20)
    assert "finish" in txt.lower()


def test_coding_plan_requires_plan_before_edits():
    txt = _render("coding_system_plan.j2")
    assert "plan" in txt.lower()
    assert "approv" in txt.lower()


def test_initial_instruction_includes_problem():
    txt = _render("initial_instruction.j2",
                  instance={"problem_statement": "FIX THE BUG X"},
                  workspace_dir_name="repo",
                  actual_workspace_path="/workspace", mode="plan")
    assert "FIX THE BUG X" in txt
