# Contributing

## Checks

```bash
make check  # validate CHANGELOG.md and changelog.d entries, and that every version matches
make test   # run the changelog script's tests
```

CI runs both on every pull request and push to `main`.

## Changelog

[`CHANGELOG.md`](CHANGELOG.md) follows [Keep a Changelog 2.0.0](https://keepachangelog.com/en/2.0.0/), with one section per `library_version`.

A PR with a user-facing change does not edit `CHANGELOG.md`. It adds one file to [`changelog.d/`](changelog.d/README.md), named `<type>-<slug>.md` for one of `added`, `changed`, `deprecated`, `removed`, `fixed`, or `security`. The whole file is the bullet, with no heading above it. One entry per file keeps PRs from conflicting. `make changelog` prints the `[Unreleased]` section with every entry file folded in.

Write for the person updating the library: which node changed, what they will notice, and why. Link the GitHub issue on the entry's last line when there is one. Start breaking changes, anything that makes a saved workflow stop working or give a different result, with `**Breaking:**`. Refactors, tests, CI, and docs-only changes get no entry. The full style guide is in [CLAUDE.md](CLAUDE.md#changelog).

`make check/changelog`, part of `make check` and CI, validates the structure of `CHANGELOG.md` and of every entry file. It never requires a PR to add an entry. That call stays with people.

## Versioning

The version is `library_version` in `griptape_nodes_lora_training_library/griptape-nodes-library.json`, mirrored in `griptape-nodes-library-cuda129.json`, `pyproject.toml`, and `uv.lock`. Bump it on `main` with one of:

```bash
make version/patch
make version/minor
make version/major
make version/set v=1.2.3
```

Each writes all four, folds the `changelog.d/` entries into a `CHANGELOG.md` section for the new version and deletes them, and commits all of it. It stops first if any of those files holds uncommitted changes, since the roll would overwrite them. Push the commit, then run `make version/publish` to tag it as `v<version>`; the compare links in `CHANGELOG.md` point at those tags.
