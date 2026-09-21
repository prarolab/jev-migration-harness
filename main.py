import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from pydantic import ValidationError

from harness import MigrationHarness
from models import AssessmentInput
from providers import DemoProviders, LiveProviders, ProviderError, require_env


ROOT = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser(description="Evidence-backed Jev + LangChain migration POC.")
    parser.add_argument("--mode", choices=["demo", "live"], default="demo")
    parser.add_argument("--strategy", choices=["hybrid", "llm-only"], default="hybrid")
    parser.add_argument("--input", type=Path, default=ROOT / "sample_input.json")
    parser.add_argument("--output", type=Path, help="New local JSON file; existing files are not overwritten.")
    parser.add_argument("--allow-external", action="store_true",
                        help="Authorize sending scoped input to TypeSafe and OpenAI in live mode.")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Experimental threshold, not a correctness or approval guarantee.")
    parser.add_argument("--max-llm-calls", type=int, default=5)
    return parser.parse_args()


def run(args) -> dict:
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    application = AssessmentInput.model_validate(raw)
    if args.output and args.output.exists():
        raise FileExistsError("Output already exists; choose a new filename.")
    if args.mode == "live" and not args.allow_external:
        raise ValueError(
            "Live mode sends scoped evidence to TypeSafe and OpenAI. "
            "Review/redact your input and supply --allow-external only with authorization."
        )
    # This POC does not permit ambient callbacks to export evidence to a third processor.
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
        os.environ[name] = "false"
    with httpx.Client(timeout=60, follow_redirects=False) as client:
        if args.mode == "demo":
            if raw != json.loads((ROOT / "sample_input.json").read_text(encoding="utf-8")):
                raise ValueError("Demo fixtures only support the bundled fictional input.")
            providers = DemoProviders()
        else:
            providers = LiveProviders(
                client,
                require_env("TYPESAFE_API_KEY") if args.strategy == "hybrid" else "",
                require_env("OPENAI_API_KEY"),
                require_env("OPENAI_MODEL"),
            )
        harness = MigrationHarness(
            providers.jev, providers.llm,
            jev_model=os.environ.get("JEV_MODEL", "jev-1.13.0"),
            threshold=args.threshold, max_llm_calls=args.max_llm_calls,
            strategy=args.strategy,
        )
        result = harness.chain.invoke(application)
        result["execution_mode"] = args.mode
        result["simulated"] = args.mode == "demo"
        result["provider_calls"] = providers.calls
    return result


def main() -> int:
    args = arguments()
    try:
        result = run(args)
        serialized = json.dumps(result, indent=2, ensure_ascii=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
            print(f"Saved {args.mode} assessment to {args.output}")
        else:
            print(serialized)
    except ValidationError as exc:
        # Pydantic's default error formatting includes rejected input values.
        print(f"Schema validation failed ({exc.error_count()} errors); no assessment published.", file=sys.stderr)
        return 1
    except (ValueError, OSError, ProviderError, httpx.HTTPError) as exc:
        print(f"Assessment failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
