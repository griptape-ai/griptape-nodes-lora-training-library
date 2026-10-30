SHELL := /bin/bash

LIBRARY_JSON := griptape_nodes_lora_training_library/griptape-nodes-library.json
# Every library JSON carries the same library_version; LIBRARY_JSON is the one it is read from.
LIBRARY_JSONS := $(LIBRARY_JSON) griptape_nodes_lora_training_library/griptape-nodes-library-cuda129.json
VERSIONED_FILES := $(LIBRARY_JSONS) pyproject.toml uv.lock CHANGELOG.md changelog.d

.PHONY: version/get
version/get: ## Get version.
	@jq -r '.metadata.library_version' $(LIBRARY_JSON)

.PHONY: version/write
version/write: ## Write version to every library JSON, pyproject.toml, and uv.lock. Usage: make version/write v=1.2.3
	@if [[ -z "$(v)" ]]; then echo "version/write requires v=<version>" >&2; exit 1; fi
	@set -e; \
	trap 'for json in $(LIBRARY_JSONS); do rm -f "$$json.tmp"; done' EXIT; \
	for json in $(LIBRARY_JSONS); do \
		jq --indent 4 --arg v "$(v)" '.metadata.library_version = $$v' "$$json" > "$$json.tmp"; \
	done; \
	for json in $(LIBRARY_JSONS); do mv "$$json.tmp" "$$json"; done
	@uv version --no-sync "$(v)"

.PHONY: version/set
version/set: changelog/assert-committed ## Set version. Usage: make version/set v=1.2.3
	@$(MAKE) --no-print-directory version/write v="$(v)"
	@$(MAKE) --no-print-directory version/commit

.PHONY: version/patch
version/patch: changelog/assert-committed ## Bump patch version.
	@CURRENT=$$($(MAKE) --no-print-directory version/get); \
	IFS='.' read -r major minor patch <<< "$$CURRENT"; \
	$(MAKE) --no-print-directory version/write v="$${major}.$${minor}.$$((patch + 1))"
	@$(MAKE) --no-print-directory version/commit

.PHONY: version/minor
version/minor: changelog/assert-committed ## Bump minor version.
	@CURRENT=$$($(MAKE) --no-print-directory version/get); \
	IFS='.' read -r major minor patch <<< "$$CURRENT"; \
	$(MAKE) --no-print-directory version/write v="$${major}.$$((minor + 1)).0"
	@$(MAKE) --no-print-directory version/commit

.PHONY: version/major
version/major: changelog/assert-committed ## Bump major version.
	@CURRENT=$$($(MAKE) --no-print-directory version/get); \
	IFS='.' read -r major minor patch <<< "$$CURRENT"; \
	$(MAKE) --no-print-directory version/write v="$$((major + 1)).0.0"
	@$(MAKE) --no-print-directory version/commit

.PHONY: version/commit
version/commit: ## Roll the changelog into the version, then commit both.
	@uv run --no-project python scripts/changelog.py roll $$($(MAKE) --no-print-directory version/get) --allow-empty
	@git commit -m "chore: bump v$$($(MAKE) --no-print-directory version/get)" -- $(VERSIONED_FILES)

# The roll overwrites CHANGELOG.md and deletes the entry files, and the bump commits whatever it left,
# so an entry still being written would ship half-finished or be lost. The version files are covered too:
# the bump commits them whole, so an unrelated edit would ride along, and a bump whose roll failed leaves
# the new version there, so bumping again from it skips a version.
.PHONY: changelog/assert-committed
changelog/assert-committed:
	@git diff --quiet HEAD -- $(VERSIONED_FILES) \
		&& test -z "$$(git ls-files --others --exclude-standard -- changelog.d)" \
		|| { echo "$(VERSIONED_FILES): one of these holds changes that are not committed."; \
			echo "Commit or stash them first. If a version bump stopped partway, undo it with:"; \
			echo "  git restore $(VERSIONED_FILES)"; exit 1; }

.PHONY: version/publish
version/publish: ## Tag the pushed version bump and push the tag. CHANGELOG.md compare links point at it.
	@set -e; \
	VERSION=$$($(MAKE) --no-print-directory version/get); \
	git fetch --quiet origin main; \
	test "$$(git log -1 --format=%s)" = "chore: bump v$$VERSION" \
		|| { echo "HEAD is not the v$$VERSION bump commit. Check out the bump on main, then publish." >&2; exit 1; }; \
	git merge-base --is-ancestor HEAD origin/main \
		|| { echo "HEAD is not on origin/main. Push the bump to main first." >&2; exit 1; }; \
	git tag "v$$VERSION"; \
	git push origin "v$$VERSION"

.PHONY: check
check: check/versions check/changelog ## Run all checks.

.PHONY: check/versions
check/versions: ## Check every library JSON, pyproject.toml, and uv.lock carry the same version.
	@set -e; \
	expected=$$($(MAKE) --no-print-directory version/get); \
	status=0; \
	for json in $(LIBRARY_JSONS); do \
		actual=$$(jq -r '.metadata.library_version' "$$json"); \
		if [[ "$$actual" != "$$expected" ]]; then echo "$$json has library_version $$actual, expected $$expected" >&2; status=1; fi; \
	done; \
	actual=$$(uv run --no-project python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"); \
	if [[ "$$actual" != "$$expected" ]]; then echo "pyproject.toml has version $$actual, expected $$expected" >&2; status=1; fi; \
	actual=$$(uv version --short --frozen); \
	if [[ "$$actual" != "$$expected" ]]; then echo "uv.lock has version $$actual, expected $$expected" >&2; status=1; fi; \
	if [[ $$status -eq 0 ]]; then echo "Versions match: $$expected"; fi; \
	exit $$status

.PHONY: check/changelog
check/changelog: ## Check CHANGELOG.md and the changelog.d entries follow Keep a Changelog.
	@uv run --no-project python scripts/changelog.py check

.PHONY: changelog
changelog: ## Print [Unreleased] as the next version will show it, changelog.d entries folded in.
	@uv run --no-project python scripts/changelog.py render

.PHONY: test
test: ## Run the changelog script's tests.
	@uv run --no-project --with pytest python -m pytest tests

.DEFAULT_GOAL := help
.PHONY: help
help: ## Print Makefile help text.
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_\/-]+:.*##/ {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
