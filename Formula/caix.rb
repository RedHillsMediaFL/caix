require "json"

class Caix < Formula
  desc "Native Apple Core AI inference server for local language models"
  homepage "https://github.com/RedHillsMediaFL/caix"
  license "MIT"
  version "0.2.0-beta"

  # Head-only until the tap has a tested 0.x release tarball.
  # Copy this file to RedHillsMediaFL/homebrew-caix/Formula/caix.rb for the tap.
  head "https://github.com/RedHillsMediaFL/caix.git", branch: "main"

  depends_on arch: :arm64
  depends_on :macos

  def install
    if OS.mac? && MacOS.version.to_s.split(".").first.to_i < 27
      odie "caix requires macOS 27+ with Apple's Core AI runtime"
    end

    coreai_frameworks = [
      "/System/Library/Frameworks/CoreAI.framework",
      "/System/Library/PrivateFrameworks/CoreAI.framework",
    ]
    unless coreai_frameworks.any? { |path| Pathname(path).directory? }
      odie "CoreAI.framework was not found; install a macOS build that ships Apple's Core AI runtime"
    end

    if File.exist?("Package.swift")
      ENV["COREAI_RUNTIME"] = "1"
      system "swift", "build", "-c", "release", "--product", "caix"
      caix_binary = ".build/release/caix"
    else
      caix_binary = "bin/caix"
      odie "release tarball is missing bin/caix" unless File.executable?(caix_binary)
    end

    libexec.install caix_binary => "caix-bin"
    pkgshare.install "web", "python", "models", "scripts", "README.md", "LICENSE"
    (pkgshare/"examples").install "docs/examples/cluster-stage-manifest.json"

    (libexec/"caix").write <<~BASH
      #!/usr/bin/env bash
      set -euo pipefail
      if [ "${1:-}" = "serve" ]; then
        shift
        exec "#{libexec}/caix-bin" serve \\
          --web "#{pkgshare}/web" \\
          --exports "${caix_exports:-$HOME/.caix/models/exports}" \\
          --registry "#{pkgshare}/models/registry.json" \\
          --convert-script "#{pkgshare}/python/converter/convert.py" \\
          "$@"
      fi
      exec "#{libexec}/caix-bin" "$@"
    BASH
    chmod 0755, libexec/"caix"
    bin.install_symlink libexec/"caix" => "caix"
  end

  def caveats
    <<~EOS
      caix requires Apple silicon and macOS 27+ with Apple's Core AI runtime.

      Verify the host:
        caix doctor

      Put converted .aimodel bundles here, or set caix_exports:
        ~/.caix/models/exports

      Start the server:
        caix serve
    EOS
  end

  test do
    assert_equal "caix #{version}", shell_output("#{bin}/caix --version").strip
    system bin/"caix", "doctor", "--no-fail"
    system bin/"caix", "cluster", "plan", "--help"
    system bin/"caix", "cluster", "join", "--help"
    assert_match("--cluster", shell_output("#{bin}/caix --help"))

    output = shell_output("#{bin}/caix cluster plan " \
                          "--manifest #{pkgshare}/examples/cluster-stage-manifest.json " \
                          "--workers main=4,mini=2 --json")
    plan = JSON.parse(output)
    runtime = plan.fetch("runtime_plan")
    boundary = plan.fetch("boundary_tensor")

    assert_equal true, plan.fetch("dry_run")
    assert_equal ["embeddings", "transformer_layers", "transformer_layers", "final_norm_head"],
                 runtime.fetch("stages").map { |stage| stage.fetch("role") }
    assert_equal 28, runtime.fetch("total_layer_count")
    assert_equal "hidden_states", boundary.fetch("name")
    assert_equal [1, -1, 1024], boundary.fetch("shape")
    assert_equal "float16", boundary.fetch("scalar_type")
    assert_equal boundary, runtime.fetch("boundary_tensor")
  end
end
