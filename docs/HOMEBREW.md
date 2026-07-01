# Homebrew

Homebrew is the default install path for MacBook validation and versioned beta releases.

Current state:

- Apple Core AI is still beta.
- caix is still beta.
- The public tap installs the packaged arm64 binary for versioned releases.
- `--HEAD` source installs are for tap development only; do not use them for MacBook validation.
- Versioned tap releases must stay below `1.0.0` while Core AI is beta.
- Do not submit to Homebrew/core until macOS 27 and Core AI are stable.

## Current Tap Path

Create `RedHillsMediaFL/homebrew-caix` with:

```text
Formula/caix.rb
```

Copy this repo's `Formula/caix.rb` there.

Current install or upgrade:

```bash
brew tap RedHillsMediaFL/caix
brew update
brew upgrade redhillsmediafl/caix/caix || brew install redhillsmediafl/caix/caix
caix --version
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

## Local MacBook Tarball Test

Before a public tap release exists, copy these package outputs to the MacBook:

```text
dist/caix-<version>-macos-arm64.tar.gz
dist/prepare-local-brew-tap.sh
dist/Formula/caix.rb
```

Then install the versioned tarball through a local tap:

```bash
./prepare-local-brew-tap.sh --tarball ./caix-<version>-macos-arm64.tar.gz
brew install redhillsmediafl/caix/caix
brew test caix
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

## MacBook Distributed POC Prep

Use the Brew-installed binary only. Do not use checkout binaries for MacBook validation.

On both Macs, upgrade or install the current versioned tap release:

```bash
brew tap RedHillsMediaFL/caix
brew update
brew upgrade redhillsmediafl/caix/caix || brew install redhillsmediafl/caix/caix
caix --version
brew test caix
caix doctor
```

If the current validation task needs the public MTP package, install it with the catalog command.
This downloads a model bundle, so do not run it from automation unless that task is assigned:

```bash
caix catalog install redhillsmediafl/rhm-qwen3-4b-mtp-caix
```

For the tiny staged hardware POC, verify the tiny manifest and installed distributed surface first:

```bash
caix_prefix="$(brew --prefix caix)"
"$caix_prefix/share/caix/scripts/check-distributed-readiness.sh" --tiny-poc \
  --tiny-manifest /path/to/qwen3-tiny-random-coreai-staged-rope-input-f16-2x1/stage-manifest.json \
  --brew-caix "$(command -v caix)"
```

Then connect the second Mac. On each Mac, start the installed server for endpoint checks:

```bash
caix_prefix="$(brew --prefix caix)"
caix serve --host 0.0.0.0 --port 1237
```

From the coordinator, verify both machines and link speed directly:

```bash
caix deploy verify \
  --endpoint <main-mac-host>:1237 --endpoint <macbook-host>:1237 \
  --min-machines 2 --speed-bytes 4194304 --min-mbps 500 --max-latency-ms 20 \
  --fail-on-warn
```

Then run the Brew surface checker with the same endpoints and tiny manifest:

```bash
"$caix_prefix/share/caix/scripts/check-brew-distributed.sh" --caix "$(command -v caix)" --ready \
  --manifest /path/to/qwen3-tiny-random-coreai-staged-rope-input-f16-2x1/stage-manifest.json \
  --endpoint <main-mac-host>:1237 --endpoint <macbook-host>:1237 \
  --min-machines 2 --speed-bytes 4194304 --min-mbps 500 --max-latency-ms 20
```

The Brew checker fails on identity, local-machine, version, latency, or link-speed warnings unless
`--allow-warnings` is passed for a diagnostic rerun. Do not use warning-allowed runs as proof. Run
the distributed smoke from the installed `caix`, not a checkout binary. Use `caix serve --cluster
... --join-timeout 120` on the coordinator and `caix cluster join ... --connect-timeout 120` on
workers so failures exit cleanly. Do not publish distributed release notes until this gate passes.

Before copying the staged bundle to the MacBook, write copy digests on the source:

```bash
"$caix_prefix/share/caix/scripts/check-stage-bundle-copy.sh" \
  --manifest /path/to/qwen3-tiny-random-coreai-staged-rope-input-f16-2x1/stage-manifest.json \
  --write /tmp/caix-stage-copy.sha256
```

After copying the bundle and digest file, verify on the MacBook:

```bash
"$caix_prefix/share/caix/scripts/check-stage-bundle-copy.sh" \
  --manifest /path/on/macbook/qwen3-tiny-random-coreai-staged-rope-input-f16-2x1/stage-manifest.json \
  --check /path/on/macbook/caix-stage-copy.sha256
```

To print exact commands for the tiny staged smoke:

```bash
"$caix_prefix/share/caix/scripts/check-tiny-cluster-smoke.sh" --caix "$(command -v caix)" \
  --manifest /path/to/qwen3-tiny-random-coreai-staged-rope-input-f16-2x1/stage-manifest.json \
  --coordinator <main-mac-host>:1237 \
  --bind-host 0.0.0.0 \
  --worker-root /path/on/macbook/qwen3-tiny-random-coreai-staged-rope-input-f16-2x1 \
  --print-commands
```
