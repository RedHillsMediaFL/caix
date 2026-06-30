# Releases

caix stays on `0.x` while Apple's Core AI runtime is beta.

Current release line:

- GitHub releases: `v0.x.y-beta` prereleases.
- Brew tap: tested `0.x` releases first; `--HEAD` remains available.
- Homebrew/core: wait for stable macOS 27/Core AI support.
- `v1.0.0` or higher: only after Core AI is no longer beta.

Before tagging:

```bash
scripts/check-version-sync.sh
scripts/check-release-version.sh v0.2.0-beta
```

For a packaged release:

```bash
scripts/package.sh 0.2.0-beta
```

The package script checks that `caix --version` matches the requested version. It also refuses
`1.0.0` or higher unless the Core AI beta gate is explicitly lifted. It also checks the
distributed Brew surface against the staged Qwen3 example manifest.

Every version bump must update all of these in the same commit:

- `Sources/PipelineCLI/BuildInfo.swift`
- `scripts/package.sh`
- `Formula/caix.rb`

`scripts/check-version-sync.sh` enforces that match.

Distributed releases must pass the Brew-installed readiness gate before cross-Mac testing:

```bash
brew tap RedHillsMediaFL/caix
brew reinstall caix
brew test caix
scripts/check-publication-gates.sh --distributed --brew-caix "$(command -v caix)"
```

The formula test must keep checking `caix cluster plan --help`, `caix cluster join --help`,
`caix deploy verify --help` advertising speed checks, top-level `caix --help` advertising
`--cluster`, exact `caix --version`, and the staged manifest plan contract.

Any release that exposes distributed inference must ship that surface through the tap formula. Use
the installed binary for Thunderbolt tests, not a loose debug build.

The tap formula supports both paths: source builds for `--HEAD`, and packaged `bin/caix` installs
for versioned releases.
