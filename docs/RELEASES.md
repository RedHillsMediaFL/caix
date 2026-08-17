# Releases

caix stays on `0.x` while Apple's Core AI runtime is beta.

Current release line:

- GitHub releases: `v0.x.y-beta` prereleases.
- Brew tap: tested `0.x` releases first; `--HEAD` remains available.
- Homebrew/core: wait for stable macOS 27/Core AI support.
- `v1.0.0` or higher: only after Core AI is no longer beta.

Before tagging:

```bash
scripts/collect-model-revisions.sh \
  --out benchmarks/revisions.tsv \
  --details benchmarks/revisions-details.tsv
scripts/check-model-revisions.sh
scripts/check-version-sync.sh
scripts/check-release-version.sh v0.2.0-beta
scripts/check-publication-gates.sh
```

The local publication gate also runs `scripts/check-release-version-contract.sh`, which fixture-tests
the Core AI beta `<1.0.0` policy, malformed-version rejection, dev-version rejection, and the explicit
stable-runtime override.
`benchmarks/revisions.tsv` is an ignored local artifact, but publication gates require it so model
tester requests and benchmark evidence stay pinned to exact Hub commits. Re-run
`scripts/collect-model-revisions.sh` immediately before release review if any published model repo was
updated.
Use `scripts/check-publication-gates.sh --strict-benchmark-gaps` for a release that publishes new
speed tables, and use the distributed command below for any distributed-inference release claim.

For a packaged release:

```bash
export TMPDIR=/Volumes/SSD/caix/.tmp/coreai-tmp
scripts/package.sh 0.2.0-beta
```

The package script checks that `caix --version` matches the requested version. It also refuses
`1.0.0` or higher unless the Core AI beta gate is explicitly lifted. It also checks the
distributed Brew surface against the staged Qwen3 example manifest. If `TMPDIR` is unset or points
at macOS system temp (`/tmp`, `/private/tmp`, or `/var/folders/...`), package staging defaults to
`.tmp/coreai-tmp` in this checkout so release builds do not spill onto the internal system temp
volume. It writes `dist/caix-<version>-macos-<arch>.tar.gz.sha256`; use that digest for the tap
formula and release notes.

Every version bump must update all of these in the same commit:

- `Sources/PipelineCLI/BuildInfo.swift`
- `scripts/package.sh`
- `Formula/caix.rb`

`scripts/check-version-sync.sh` enforces that match.
`scripts/check-package-contract.sh` enforces the package staging/temp and SHA-256 sidecar invariants
without building a tarball; publication gates run it.

Distributed releases must pass the Brew-installed readiness gate before cross-Mac testing:

```bash
brew tap RedHillsMediaFL/caix
brew reinstall caix
brew test caix
scripts/check-publication-gates.sh --distributed --strict-evidence --brew-caix "$(command -v caix)"
```

`--strict-evidence` keeps final distributed release review from passing on untracked local parity
summaries or raw logs.

The formula test must keep checking `caix cluster plan --help`, `caix cluster join --help`,
`caix deploy verify --help` advertising speed checks, top-level `caix --help` advertising
`--cluster`, exact `caix --version`, and the staged manifest plan contract.

Any release that exposes distributed inference must ship that surface through the tap formula. Use
the installed binary for Thunderbolt tests, not a loose debug build.

The tap formula supports both paths: source builds for `--HEAD`, and packaged `bin/caix` installs
for versioned releases.
