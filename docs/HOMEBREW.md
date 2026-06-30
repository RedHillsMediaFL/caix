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
4. Build `caix-<version>-macos-arm64.tar.gz` with `scripts/package.sh <version>`.
5. Cut a release tag below `v1.0.0`.
6. Upload the tarball.
7. Update the tap formula with the release URL and SHA-256. The formula can install the packaged
   `bin/caix` binary for a versioned release or build from source for `--HEAD`.
8. Test:

```bash
brew audit --strict --online caix
brew install caix
brew test caix
caix doctor
caix_prefix="$(brew --prefix caix)"
"$caix_prefix/share/caix/scripts/check-brew-distributed.sh" --caix "$(command -v caix)" --ready \
  --manifest "$caix_prefix/share/caix/examples/cluster-stage-manifest.json"
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
brew test caix
scripts/check-publication-gates.sh --distributed --brew-caix "$(command -v caix)"
```

Then connect the second Mac and verify the link from the installed `caix`:

```bash
caix_prefix="$(brew --prefix caix)"
"$caix_prefix/share/caix/scripts/check-brew-distributed.sh" --caix "$(command -v caix)" --ready \
  --manifest /path/to/qwen3-tiny-random-coreai-staged-rope-input-f16-2x1/stage-manifest.json \
  --endpoint <main-mac-host>:1237 --endpoint <macbook-host>:1237 \
  --min-machines 2 --speed-bytes 4194304 --min-mbps 500 --max-latency-ms 20
```

If that warns, fix wiring before running the cluster smoke. Run the distributed smoke from the
installed `caix`, not a checkout binary.
Do not publish distributed release notes until this gate passes.
