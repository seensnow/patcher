"""Ask OpenHands to fix a failing pytest file and verify the repair."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pytest import Session, TestReport


@dataclass
class PytestReport:
    status: str = ""
    collected: list[str] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def pytest_collection_finish(self, session: Session) -> None:
        self.collected = [item.nodeid for item in session.items]

    def pytest_xdist_node_collection_finished(self, ids: list[str]) -> None:
        self.collected = ids

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        if report.failed:
            self.failed.append(report.nodeid)
        elif report.skipped:
            self.skipped.append(report.nodeid)
        elif report.when == "call" and report.passed:
            self.passed.append(report.nodeid)

    def all_passed(self) -> bool:
        return (
            self.status == "OK"
            and bool(self.collected)
            and not self.failed
            and not self.skipped
            and Counter(self.passed) == Counter(self.collected)
        )


def run_pytest_worker(report_path: str, test_file: str) -> int:
    import pytest

    report = PytestReport()
    pytest.hookimpl(optionalhook=True)(
        PytestReport.pytest_xdist_node_collection_finished
    )
    status = pytest.main(["-ra", "--tb=short", "--color=no", "--", test_file], [report])
    report.status = pytest.ExitCode(status).name
    Path(report_path).write_text(json.dumps(asdict(report)), encoding="utf-8")
    return status


def run_pytest(
    project: Path, test_file: Path, scratch: Path
) -> tuple[PytestReport, str]:
    report_path = scratch / "pytest.json"
    report_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-c",
        "import runpy, sys; "
        "worker = runpy.run_path(sys.argv[1])['run_pytest_worker']; "
        "raise SystemExit(worker(*sys.argv[2:]))",
        str(Path(__file__).resolve()),
        str(report_path),
        str(test_file),
    ]
    result = subprocess.run(
        command,
        cwd=project,
        # Read fresh source even for same-size edits within one second.
        env=os.environ
        | {
            "PYTHONPYCACHEPREFIX": str(scratch / "pycache"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    print(result.stdout, end="", flush=True)
    if not report_path.is_file():
        raise RuntimeError(
            f"pytest exited before reporting results (exit {result.returncode})."
        )
    report = PytestReport(**json.loads(report_path.read_text(encoding="utf-8")))
    expected_exit = {"OK": 0, "TESTS_FAILED": 1, "NO_TESTS_COLLECTED": 5}
    if expected_exit.get(report.status) != result.returncode:
        raise RuntimeError(
            f"pytest could not complete the run ({report.status}, "
            f"exit {result.returncode})."
        )
    return report, result.stdout


def check_test_file(test_file: Path, original: bytes) -> None:
    try:
        current = test_file.read_bytes()
    except OSError as error:
        raise RuntimeError(
            f"Test file is missing or unreadable: {test_file} ({error})"
        ) from error
    if current != original:
        raise RuntimeError(f"Test file was modified: {test_file}")


def repair(project: Path, test_file: Path, output: str) -> float:
    from openhands.sdk import LLM, Agent, Conversation, Tool
    from openhands.tools.file_editor import FileEditorTool

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Set LLM_API_KEY before running a repair.")
    llm = LLM(
        model=os.getenv("LLM_MODEL", "gpt-5.5"),
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL"),
    )
    agent = Agent(llm=llm, tools=[Tool(name=FileEditorTool.name)])
    conversation = Conversation(agent=agent, workspace=str(project))
    try:
        conversation.send_message(
            f"Fix the implementation in {project} so the pytest tests in {test_file} "
            "pass. Read the tests to understand the expected behavior. Edit only "
            "implementation files. Leave tests, fixtures, and pytest configuration "
            "unchanged. Patcher will rerun pytest after you finish.\n\n"
            f"Initial pytest output:\n{output}"
        )
        conversation.run()
        return conversation.conversation_stats.get_combined_metrics().accumulated_cost
    finally:
        conversation.close()


def patch(project: Path, test_file: Path, *, demo: bool = False) -> None:
    original = test_file.read_bytes()
    with TemporaryDirectory(prefix="patcher-") as directory:
        scratch = Path(directory)
        before, output = run_pytest(project, test_file, scratch)
        check_test_file(test_file, original)
        if not before.collected:
            raise RuntimeError(f"pytest collected no tests from {test_file}.")
        if before.all_passed():
            print("Already passing")
            return
        if not before.failed:
            raise RuntimeError(
                "pytest reported no failures, but some tests did not pass."
            )

        try:
            cost = repair(project, test_file, output)
        except Exception as error:
            raise RuntimeError(f"OpenHands repair failed: {error}") from error
        finally:
            check_test_file(test_file, original)

        after, _ = run_pytest(project, test_file, scratch)
        check_test_file(test_file, original)
        original_cases = Counter(before.collected)
        verified_cases = Counter(after.collected)
        if original_cases != verified_cases:
            removed = list((original_cases - verified_cases).elements())
            added = list((verified_cases - original_cases).elements())
            raise RuntimeError(
                f"Collected tests changed. Removed: {removed}. Added: {added}."
            )
        if after.failed:
            raise RuntimeError("Tests still failing after repair.")
        if after.skipped:
            raise RuntimeError(
                f"Tests skipped after repair: {', '.join(after.skipped)}"
            )
        if not after.all_passed():
            raise RuntimeError(
                "Some collected tests did not run to completion after repair."
            )
        if demo:
            print(f"EXAMPLE_COST: {cost}")
        print("Verified")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="project directory")
    parser.add_argument(
        "test_file", type=Path, help="pytest file, relative to the project"
    )
    parser.epilog = "With no arguments, run a repair demo in a temporary directory."
    try:
        if len(sys.argv) == 1:
            with TemporaryDirectory(prefix="patcher-demo-") as directory:
                project = Path(directory)
                (project / "arithmetic.py").write_text(
                    "def add(a, b):\n    return a - b\n", encoding="utf-8"
                )
                test_file = project / "test_arithmetic.py"
                test_file.write_text(
                    "from arithmetic import add\n\n"
                    "def test_add():\n    assert add(2, 3) == 5\n",
                    encoding="utf-8",
                )
                patch(project, test_file, demo=True)
            return 0

        args = parser.parse_args()
        project = args.project.resolve()
        test_file = project / args.test_file
        if not project.is_dir():
            parser.error(f"project directory does not exist: {project}")
        if not test_file.resolve().is_relative_to(project) or not test_file.is_file():
            parser.error(
                f"test file must be an existing file inside {project}: {test_file}"
            )
        patch(project, test_file)
    except (OSError, RuntimeError) as error:
        print(f"patcher: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
