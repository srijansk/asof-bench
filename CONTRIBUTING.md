# Contributing to asof-bench

asof-bench v0.1 is a pre-registered benchmark: [PREREGISTRATION.md](PREREGISTRATION.md) fixes its questions, gold, scoring and entrants. The registration is PREREGISTRATION.md, [labelling/GUIDELINES.md](labelling/GUIDELINES.md), every module in `src/asofbench` and the files in data/v0.1, frozen by the annotated tag `prereg-v0.1.2`, and every change to them after the freeze is recorded in [DEVIATIONS.md](DEVIATIONS.md) with its date and reason. So open an issue before a pull request that touches a registered file: the change may have to wait for the next version. Everything else, the README, the pages of [docs/](docs/), the examples, the tests and the repository's own files, can change freely.

## Reporting problems

Open an issue at https://github.com/srijansk/asof-bench/issues. For a suspected wrong gold answer, give the item id, the answer you expect, and the repository, path and commit that show it; the [gold-error template](.github/ISSUE_TEMPLATE/gold-error.md) asks for exactly these. An item's as-of commit in each repository is its gold record's `as_of_commits`: the last first-parent commit whose recorded time is at or before the end of the item's date ([section 3](PREREGISTRATION.md#3-recorded-time-and-as-of)). Read the files at that commit, not with git's own date filters. For an F1 item, the gold record's `source` names the repository, path and commit of the file whose header set the answer, so `git -C ../asofbench-clones/<repo> show <commit>:<path> | grep -m1 -i '^status:'` prints the header the answer was read from ([docs/data-format.md](docs/data-format.md#goldjsonl)). `git log --until` (or `--before`) reads raw committer dates, and in python/peps, where a merge committed in 2012 carries a 2002 committer date, `git log --first-parent --until=<D>T23:59:59Z -1` gives a commit other than the as-of commit for every date D from 2002-01-08 to 2012-09-22, and on five single days, 2013-08-14, 2013-10-22, 2015-05-20, 2015-08-22 and 2016-04-14; and `<branch>@{<date>}` reads your clone's reflog, not the project's history, so in a fresh clone it gives the branch as you cloned it. The registration is frozen: a confirmed error in a registered file is recorded in [DEVIATIONS.md](DEVIATIONS.md) with its date and reason, and affected results are reported both as registered and as amended.

## Reviewing an adapter

If you maintain one of the systems that v0.1 enters, you are invited to review its adapter and configuration before any scored run: open an issue, as [Entrants and outside entries](docs/entering-a-system.md#entrants-and-outside-entries) describes.

## Development

```bash
uv run --frozen --extra dev pytest
ASOFBENCH_CLONES=../asofbench-clones uv run --frozen --extra dev pytest
uv run --frozen --extra dev pytest systems/tests
uv run --frozen --extra dev --with anthropic==0.111.0 --with httpx==0.28.1 python -m pytest systems/tests/test_llm_client.py
uv run --frozen --extra dev ruff check src tests examples systems
```

The third and fourth commands run the tests of `systems/` as CI does: first on the repository's own environment, where the tests of the systems' Anthropic client skip, and then those tests with the SDK and httpx at the versions that the Graphiti image's lock pins, through `python -m pytest`, whose interpreter sees the `--with` packages. The few that also need graphiti-core run inside the Graphiti image ([systems/graphiti/README.md](systems/graphiti/README.md#tests)). With `UV_OFFLINE=1` before a command, uv installs nothing from the network, which works once its cache holds what the command needs.

`--extra dev` installs the test tools (pytest, pytest-cov and ruff), and `--frozen` installs them as uv.lock pins them. The unit tests build small git repositories of their own; the corpus tests, marked `corpus`, need `ASOFBENCH_CLONES` set to the directory that holds the three clones, and are skipped without it. With the clones, the suite takes five to ten minutes, depending on the machine. In code comments and docstrings, "spec" means PREREGISTRATION.md, so "spec section 9" is its section 9.

Release artifacts, the source distribution and the wheel, are built only from a fresh clone of the tag (`uv build` there). The source distribution holds the repository's tracked top-level files and directories, which `[tool.hatch.build.targets.sdist] only-include` in pyproject.toml lists, so a build from a working checkout cannot ship an untracked top-level file; an untracked file inside one of those directories would still ship unless .gitignore covers it, which is why a release is built from a fresh clone. The package's metadata gives its licence as `MIT AND CC0-1.0`, with LICENSE and LICENSE-DATA as its licence files, since the source distribution ships data/v0.1.

Some tests check the documents rather than the code. The document tests in tests/test_build.py check that the README, the pages of docs/ and the registration state each rule as the scorer applies it, and quote the scorer's own words; tests/test_docs.py checks that every link between the Markdown files reaches its file and heading, and that docs/data-format.md names every field the data files hold. Editing a document can therefore fail a test: move a guarded passage together with the test that reads it.

[docs/code-map.md](docs/code-map.md) maps the modules to the registration's sections and to the commands. The test names state the rules each module enforces: `uv run --frozen --extra dev pytest --collect-only -q tests/test_scoring.py` lists the scorer's. Run `ruff check`, not `ruff format`, which would rewrite the frozen modules.
