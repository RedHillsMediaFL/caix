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
`1.0.0` or higher unless the Core AI beta gate is explicitly lifted.

Every version bump must update all of these in the same commit:

- `Sources/PipelineCLI/BuildInfo.swift`
- `scripts/package.sh`
- `Formula/caix.rb`

`scripts/check-version-sync.sh` enforces that match.

Distributed inference releases need a Brew install check before cross-Mac testing:

```bash
brew tap RedHillsMediaFL/caix
brew reinstall caix
scripts/check-brew-distributed.sh
```

Use that installed binary for Thunderbolt tests, not a loose debug build.
