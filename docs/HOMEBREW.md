# Homebrew

Homebrew should be the default install path once it is tested and verified.

Current state:

- Apple Core AI is still beta.
- caix is still beta.
- The formula is HEAD-only for the tap.
- Versioned tap releases must stay below `1.0.0` while Core AI is beta.
- Do not submit to Homebrew/core until macOS 27 and Core AI are stable.

## Current Tap Path

Create `RedHillsMediaFL/homebrew-caix` with:

```text
Formula/caix.rb
```

Copy this repo's `Formula/caix.rb` there.

Current install:

```bash
brew tap RedHillsMediaFL/caix
brew install --HEAD caix
caix doctor
```

## Release Path

For a verified `0.x` release:

1. Update the version in `Sources/PipelineCLI/BuildInfo.swift`, `scripts/package.sh`, and `Formula/caix.rb`.
2. Run `scripts/check-version-sync.sh`.
3. Run `scripts/check-release-version.sh v0.2.0-beta`.
4. Cut a release tag below `v1.0.0`.
5. Build and upload `caix-<version>-macos-arm64.tar.gz`.
6. Update the tap formula with the release URL and SHA-256.
7. Test:

```bash
brew audit --strict --online caix
brew install caix
brew test caix
caix doctor
```

When those pass, document this as the first install command:

```bash
brew tap RedHillsMediaFL/caix
brew install caix
caix doctor
```

## Homebrew/Core Later

Wait until all are true:

- macOS 27 is public and stable.
- Apple's Core AI runtime is public and stable.
- caix has a `1.0.0` or newer tag.
- `caix --version` reports the tag version.
- `caix doctor` passes on a clean supported Mac.
- The formula builds from source or uses the packaging Homebrew accepts at that time.
- `brew audit --strict --online caix`, `brew test caix`, and the bottle build pass.

Then open a PR against `Homebrew/homebrew-core` adding `Formula/c/caix.rb`, or request missing macOS
27/Core AI formula support first if Homebrew does not yet expose it.

Keep the formula blunt: what it installs, required macOS/Core AI version, and how to run `caix
doctor`. No marketing copy.

## Distributed Testing

When distributed inference is ready for Thunderbolt testing, test it through the Brew-installed
binary first:

```bash
brew tap RedHillsMediaFL/caix
brew reinstall caix
scripts/check-publication-gates.sh --distributed --brew-caix "$(command -v caix)"
```

Then connect the second Mac and run the distributed smoke from the installed `caix`.
Do not publish distributed release notes until this gate passes.
