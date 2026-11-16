"""Check that every canonical recipe still parses against openadmet-models.

For each recipe this loads the YAML into an ``AnvilSpecification`` and builds the
workflow, which constructs every featurizer, model, splitter, trainer and
evaluator and so validates their parameters. No data is loaded and nothing is
trained, so recipes whose data is not vendored in this repo still pass.

Also checks that each ``*.sbatch`` launches a recipe that sits next to it.

Usage: python scripts/validate_recipes.py [recipe ...]
"""

import contextlib
import logging
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RECIPES = REPO / "canonical_recipes"

# Model directories written by a previous ``anvil`` run, not recipes
SKIP_DIR_SUFFIX = "_model_dir"

# Recipes that read an artifact an earlier recipe writes, so they cannot build
# until that one has run. Keyed by recipe path, valued by the artifact name; a
# build failure is skipped only when its message names that artifact.
NEEDS_UPSTREAM_ARTIFACT = {
    "canonical_recipes/transfer_learning/tabicl_pec50/tabicl_pec50.yaml": (
        "chemprop_log2fc_model"
    ),
}


def find_recipes() -> list[Path]:
    """Return every recipe YAML under canonical_recipes, skipping run outputs."""
    return sorted(
        p
        for p in RECIPES.rglob("*.yaml")
        if not any(part.endswith(SKIP_DIR_SUFFIX) for part in p.parts)
    )


def check_recipe(path: Path) -> str | None:
    """Build the workflow for one recipe; return an error message or None.

    Runs from the recipe's own directory, as its sbatch script does, so paths
    such as a fine-tuning ``param_path`` resolve the way they will in practice.
    A return value starting with ``skip:`` is reported but does not fail.
    """
    from openadmet.models.anvil.specification import AnvilSpecification

    try:
        with contextlib.chdir(path.parent):
            AnvilSpecification.from_recipe(str(path)).to_workflow()
    except Exception as exc:  # noqa: BLE001, report every failure, not just the first
        message = f"{type(exc).__name__}: {exc}"
        artifact = NEEDS_UPSTREAM_ARTIFACT.get(str(path.relative_to(REPO)))
        if artifact and artifact in message:
            return f"skip: needs {artifact}, written by an earlier recipe"
        return message
    return None


def check_sbatch(path: Path) -> str | None:
    """Confirm the recipe an sbatch script launches exists beside it."""
    match = re.search(r"--recipe-path\s+(\S+)", path.read_text())
    if match is None:
        return "no `openadmet anvil --recipe-path` line"
    if not (path.parent / match.group(1)).is_file():
        return f"launches {match.group(1)}, which does not exist"
    return None


def main(argv: list[str]) -> int:
    logging.disable(logging.CRITICAL)
    recipes = [Path(a).resolve() for a in argv] or find_recipes()
    sbatches = [] if argv else sorted(RECIPES.rglob("*.sbatch"))

    failures = skips = 0
    for path, check in [(r, check_recipe) for r in recipes] + [
        (s, check_sbatch) for s in sbatches
    ]:
        error = check(path)
        rel = path.relative_to(REPO)
        if error is None:
            print(f"ok    {rel}")
        elif error.startswith("skip:"):
            skips += 1
            print(f"skip  {rel}\n      {error.removeprefix('skip: ')}")
        else:
            failures += 1
            print(f"FAIL  {rel}\n      {error}")

    total = len(recipes) + len(sbatches)
    print(f"\n{total - failures - skips} passed, {skips} skipped, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
