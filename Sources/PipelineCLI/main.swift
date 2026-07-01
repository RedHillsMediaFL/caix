import CoreAIServer
import PipelineRuntime
import Darwin
import Dispatch
import Foundation
import MachineStats

// Minimal CLI entry.
//   stats                              prints a machine snapshot as JSON
//   run --model <bundle> --prompt ...  native Core AI text generation
let args = CommandLine.arguments
let command = args.count > 1 ? args[1] : "stats"

switch command {
case "-v", "--version", "version":
    printVersion()

case "doctor":
    doctorCommand(Array(args.dropFirst(2)))

case "stats":
    let snap = MachineStats.snapshot()
    let enc = JSONEncoder()
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    if let data = try? enc.encode(snap), let s = String(data: data, encoding: .utf8) {
        print(s)
    }

case "run":
    runCommand(Array(args.dropFirst(2)))

case "chat", "tui":
    chatTUICommand(Array(args.dropFirst(2)))

case "dashboard":
    dashboardCommand(Array(args.dropFirst(2)))

case "serve":
    serveCommand(Array(args.dropFirst(2)))

case "inspect":
    inspectCommand(Array(args.dropFirst(2)))

case "bench":
    benchCommand(Array(args.dropFirst(2)))

case "catalog":
    catalogCommand(Array(args.dropFirst(2)))

case "cluster":
    clusterCommand(Array(args.dropFirst(2)))

case "deploy":
    deployCommand(Array(args.dropFirst(2)))

case "eagle":
    eagleCommand(Array(args.dropFirst(2)))

case "-h", "--help", "help":
    printUsage()

default:
    FileHandle.standardError.write(Data("unknown command: \(command)\n".utf8))
    printUsage()
    exit(2)
}

// MARK: - run subcommand

func printUsage() {
    let usage = """
        caix — native Apple Core AI serving pipeline

        USAGE:
          caix --version
          caix doctor [--no-fail]
          caix stats
          caix run --model <bundle-dir> --prompt "..." [options]
          caix chat [--endpoint http://127.0.0.1:1237] [--shell ask|on|off]
          caix tui [--endpoint http://127.0.0.1:1237] [--shell ask|on|off]
          caix dashboard [--endpoint http://127.0.0.1:1237]
          caix inspect --model <bundle-dir>
          caix bench --model <bundle-dir> [options]
          caix catalog <owner/search|collection-slug> [options]
          caix cluster plan --manifest <stage-manifest.json> [options]
          caix cluster join --coordinator <host:port> --manifest <stage-manifest.json> --stage <stage-dir> [options]
          caix deploy verify --endpoint <host[:port]|url> --endpoint <host[:port]|url> [options]
          caix serve [--port 1237] [--host 127.0.0.1]
          caix serve --cluster <stage-manifest.json> [options]

        serve OPTIONS:
          --port <N>             Listen port (default: 1237)
          --host <H>             Bind host (default: 127.0.0.1)
          --exports <dir>        Exported bundles dir (default: ~/.caix/models/exports)
          --registry <path>      Model registry JSON (default: ./models/registry.json)
          --web <dir>            Dashboard web dir (default: ./web)
          --convert-script <p>   convert.py path (default: ./python/converter/convert.py)
          --python <exe>         Python executable for conversion (default: python3)
          --stats-file <path>     Persistent usage stats JSON (default: ~/.caix/usage.json)
          --prewarm <model|smallest|all|off>
                                  Warm model before listening (default: smallest)
          --no-prewarm            Start serving without first-request compile warmup
          --cluster <manifest>    Stage manifest for distributed coordinator mode
          --remote-stage <id>     Remote stage id; repeatable (default: all transformer stages)
          --prompt-tokens <list>  Comma-separated token ids for a staged POC request
          --max-tokens <N>        Generated token count for --cluster (default: 1)
          --kv-capacity <N>       KV cache capacity for --cluster
          --max-context <N>       Max context length for --cluster (default: 2048)
          --join-timeout <s>      Seconds to wait for remote stages in --cluster
          --once                  Run one cluster request or readiness check, then exit
          --no-eagle              Disable the built-in EAGLE/MTP serve model
          --eagle-name <name>     Served name for the EAGLE/MTP model
          --eagle-target <dir>    EAGLE target .aimodel bundle
          --eagle-draft <dir>     EAGLE draft .aimodel bundle
          --eagle-unrolled <dir>  Optional unrolled EAGLE draft .aimodel bundle
          --eagle-tokenizer <dir> Tokenizer directory for the EAGLE/MTP model
          --eagle-vocab <N>      EAGLE/MTP vocabulary size (default: 262144)
          --eagle-backbone <N>   EAGLE/MTP hidden size (default: 2816)
          --eagle-hidden-size <N>
                                  Alias for --eagle-backbone
          --eagle-sliding-window <N>
                                  EAGLE/MTP sliding KV window (default: 1024)
          --eagle-max-context <N>
                                  EAGLE/MTP max context length (default: 4096)
          --verbose              Emit per-request diagnostics to stderr

        run OPTIONS:
          --model <dir>          Exported .aimodel bundle directory (required)
          --draft <dir>          Draft .aimodel bundle for speculative decoding (optional)
          --draft-tokens <K>     Draft tokens proposed per step (default: 4; needs --draft)
          --prompt <text>        Prompt text (required)
          --max-tokens <N>       Max tokens to generate (default: 64)
          --temperature <t>      Sampling temperature; 0 = greedy (default: 0)
          --top-k <K>            Top-K filter (temperature > 0)
          --top-p <P>            Top-P / nucleus filter (temperature > 0)
          --seed <S>             RNG seed for reproducible sampling
          --raw                  Skip the chat template (raw completion)
          --kv-capacity <N>      Fixed KV cache capacity override (auto-floored per model)
          --verbose              Emit timing/diagnostics to stderr

        chat / tui OPTIONS:
          --endpoint <url>       caix server endpoint (default: http://127.0.0.1:1237)
          --model <name>         Served model id (default: first /v1/models result)
          --shell <mode>         Shell tool access: ask, on, off (default: ask)
          --cwd <dir>            Shell working directory (default: current directory)
          --max-tokens <N>       Max response tokens (default: 1024)
          --temperature <t>      Sampling temperature (default: 0.7)
          --system <text>        System prompt

        bench OPTIONS:
          --model <dir>          Exported .aimodel bundle directory (required)
          --seqs <A,B,C>         Forward sequence lengths (default: 1,2,4,7,1)
          --warmup <N>           Warmup forwards per sequence length (default: 4)
          --iters <N>            Measured forwards per sequence length (default: 10)
                                  This is a forward micro-benchmark, not decode tok/s.

        catalog OPTIONS:
          --limit <N>            Max repos to show (default: 25)
          --json                 Emit machine-readable JSON

        cluster plan OPTIONS:
          --manifest <path>      Local stage manifest JSON
          --model <bundle-dir>   Read cluster.stages from bundle metadata.json
          --worker <name=GB>     Worker memory budget; repeat or use --workers a=64,b=32
          --workers <list>       Comma-separated worker memory budgets
          --kv-capacity <N>      Include estimated KV cache for N tokens
          --headroom-gb <GB>     Reserve per-worker OS/runtime headroom
          --dry-run              Accepted for clarity; planning is dry-run only
          --json                 Emit machine-readable JSON

        cluster join OPTIONS:
          --coordinator <host:port>
                                  Coordinator address for this worker
          --stage <dir>           Local staged .aimodel bundle directory
          --listen <host:port>    Worker listen address (default: 127.0.0.1:0)

        deploy verify OPTIONS:
          --endpoint, -e <target>  caix server endpoint; repeatable
          --endpoints <list>       Comma-separated endpoints
          --min-machines <N>       Distinct reachable machine identities required (default: 2)
          --timeout <seconds>      Per-endpoint HTTP timeout (default: 2)
          --path <path>            Probe path when endpoint has no path (default: /api/server)
          --speed-bytes <N>        Download bytes per endpoint (default: 4194304)
          --min-mbps <N>           Warn below this download rate (default: 500)
          --max-latency-ms <N>     Warn above this /api/server latency (default: 20)
          --no-speed-test          Skip download speed probe
          --fail-on-warn           Exit non-zero for blocker warnings
          --json                   Emit machine-readable JSON

        DIFFUSION (auto-detected from bundle metadata kind/diffusion block):
          run routes diffusion bundles to the host denoise loop (random-canvas init →
          entropy/MI-bound accept+renoise with self-conditioning → adaptive stop → block
          commit). --temperature/--top-k/--top-p/--draft do not apply.
          env COREAI_DIFFUSION_MAX_STEPS=<N>  cap denoise steps (smoke/debug; default: metadata)
          env COREAI_DIFFUSION_CANVAS=<N>     cap canvas length (smoke/debug; default: metadata)
        """
    print(usage)
}

func printVersion() {
    print("caix \(CaixBuildInfo.version)")
}

func doctorCommand(_ argv: [String]) {
    var noFail = false
    var i = 0
    while i < argv.count {
        switch argv[i] {
        case "--no-fail":
            noFail = true
        case "-h", "--help":
            print(
                """
                USAGE:
                  caix doctor [--no-fail]

                Checks Apple silicon, macOS 27+, CoreAI.framework, and a runtime-linked build.
                """
            )
            exit(0)
        default:
            FileHandle.standardError.write(Data("unknown doctor option: \(argv[i])\n".utf8))
            exit(2)
        }
        i += 1
    }

    struct Check {
        let name: String
        let ok: Bool
        let required: Bool
        let detail: String
    }

    let os = ProcessInfo.processInfo.operatingSystemVersion
    let osString = "\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)"
    let arch = machineArchitecture()
    let coreAIFramework = firstExistingPath([
        "/System/Library/Frameworks/CoreAI.framework",
        "/System/Library/PrivateFrameworks/CoreAI.framework",
    ])
    let swiftPath = findExecutable("swift")

    let checks = [
        Check(name: "caix version", ok: true, required: false, detail: CaixBuildInfo.version),
        Check(name: "Apple silicon", ok: arch == "arm64", required: true, detail: arch),
        Check(
            name: "macOS \(CaixBuildInfo.requiredMacOSMajor)+",
            ok: os.majorVersion >= CaixBuildInfo.requiredMacOSMajor,
            required: true,
            detail: osString),
        Check(
            name: "CoreAI.framework",
            ok: coreAIFramework != nil,
            required: true,
            detail: coreAIFramework ?? "not found"),
        Check(
            name: "runtime-linked binary",
            ok: CoreAIPipeline.isLinked,
            required: true,
            detail: CoreAIPipeline.isLinked ? "linked" : "not linked; rebuild with COREAI_RUNTIME=1"),
        Check(
            name: "Swift toolchain",
            ok: swiftPath != nil,
            required: false,
            detail: swiftPath ?? "not found; only needed for source builds/conversion"),
    ]

    print("caix doctor")
    var failures = 0
    for check in checks {
        let label: String
        if check.ok {
            label = "ok"
        } else if check.required {
            label = "fail"
            failures += 1
        } else {
            label = "warn"
        }
        print("[\(label)] \(check.name): \(check.detail)")
    }
    if failures > 0 && !noFail {
        exit(1)
    }
}

func machineArchitecture() -> String {
    var systemInfo = utsname()
    uname(&systemInfo)
    return withUnsafePointer(to: &systemInfo.machine) {
        $0.withMemoryRebound(to: CChar.self, capacity: 1) {
            String(cString: $0)
        }
    }
}

func firstExistingPath(_ paths: [String]) -> String? {
    let fm = FileManager.default
    return paths.first { fm.fileExists(atPath: $0) }
}

func findExecutable(_ name: String) -> String? {
    let fm = FileManager.default
    let path = ProcessInfo.processInfo.environment["PATH"] ?? ""
    for dir in path.split(separator: ":") {
        let candidate = "\(dir)/\(name)"
        if fm.isExecutableFile(atPath: candidate) {
            return candidate
        }
    }
    return nil
}

// MARK: - inspect subcommand

func inspectCommand(_ argv: [String]) {
    var model: String?
    var i = 0
    while i < argv.count {
        switch argv[i] {
        case "--model", "-m":
            i += 1
            guard i < argv.count else {
                FileHandle.standardError.write(Data("error: --model needs a value\n".utf8))
                exit(2)
            }
            model = argv[i]
        case "-h", "--help":
            printUsage()
            exit(0)
        default:
            FileHandle.standardError.write(Data("unknown inspect option: \(argv[i])\n".utf8))
            exit(2)
        }
        i += 1
    }
    guard let modelPath = model else {
        FileHandle.standardError.write(Data("error: inspect requires --model <bundle>\n".utf8))
        exit(2)
    }

    let semaphore = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var exitCode: Int32 = 0
    Task {
        defer { semaphore.signal() }
        do {
            let description = try await CoreAIPipeline.inspectBundle(modelPath: modelPath)
            FileHandle.standardOutput.write(Data((description + "\n").utf8))
        } catch {
            FileHandle.standardError.write(Data("inspect error: \(error)\n".utf8))
            exitCode = 1
        }
    }
    while semaphore.wait(timeout: .now()) == .timedOut {
        RunLoop.main.run(mode: .default, before: Date().addingTimeInterval(0.02))
    }
    exit(exitCode)
}

func runCommand(_ argv: [String]) {
    var model: String?
    var draft: String?
    var draftTokens = 4
    var prompt: String?
    var maxTokens = 64
    var temperature = 0.0
    var topK: Int?
    var topP: Double?
    var seed: UInt64?
    var applyChatTemplate = true
    var kvCapacity: Int?
    var verbose = false

    func fail(_ msg: String) -> Never {
        FileHandle.standardError.write(Data("error: \(msg)\n".utf8))
        exit(2)
    }

    var i = 0
    func value(_ flag: String) -> String {
        i += 1
        guard i < argv.count else { fail("missing value for \(flag)") }
        return argv[i]
    }
    func intValue(_ flag: String) -> Int {
        guard let v = Int(value(flag)) else { fail("invalid \(flag) (expected integer)") }
        return v
    }
    func doubleValue(_ flag: String) -> Double {
        guard let v = Double(value(flag)) else { fail("invalid \(flag) (expected number)") }
        return v
    }

    while i < argv.count {
        let arg = argv[i]
        switch arg {
        case "--model", "-m": model = value(arg)
        case "--draft", "-d": draft = value(arg)
        case "--draft-tokens": draftTokens = intValue(arg)
        case "--prompt", "-p": prompt = value(arg)
        case "--max-tokens", "-n": maxTokens = intValue(arg)
        case "--temperature", "-t": temperature = doubleValue(arg)
        case "--top-k": topK = intValue(arg)
        case "--top-p": topP = doubleValue(arg)
        case "--seed":
            guard let s = UInt64(value(arg)) else { fail("invalid --seed") }
            seed = s
        case "--kv-capacity": kvCapacity = intValue(arg)
        case "--raw", "--no-chat-template": applyChatTemplate = false
        case "--verbose", "-v": verbose = true
        case "-h", "--help": printUsage(); exit(0)
        default: fail("unknown option: \(arg)")
        }
        i += 1
    }

    guard let modelPath = model else { fail("--model is required") }
    guard let promptText = prompt else { fail("--prompt is required") }

    if !CoreAIPipeline.isLinked {
        FileHandle.standardError.write(
            Data(
                "note: Core AI runtime not linked — rebuild with COREAI_RUNTIME=1 (Xcode 27 / macOS 27).\n"
                    .utf8))
    }

    let options = CoreAIPipeline.Options(
        maxTokens: maxTokens,
        temperature: temperature,
        topK: topK,
        topP: topP,
        applyChatTemplate: applyChatTemplate,
        kvCapacity: kvCapacity,
        seed: seed,
        verbose: verbose)

    // Bridge the async runtime onto the blocking CLI entry point.
    let draftPath = draft
    let semaphore = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var exitCode: Int32 = 0
    Task {
        defer { semaphore.signal() }
        do {
            let onToken: (String) -> Void = { delta in
                FileHandle.standardOutput.write(Data(delta.utf8))
            }
            if CoreAIPipeline.isDiffusionBundle(modelPath: modelPath) {
                // Block-diffusion bundle (stateless bidirectional forward + host denoise loop).
                let result = try await CoreAIPipeline.runDiffusion(
                    modelPath: modelPath,
                    prompt: promptText,
                    options: options,
                    onToken: onToken)
                FileHandle.standardOutput.write(Data("\n".utf8))
                // The denoise mechanics are the headline — always report them.
                let summary = String(
                    format:
                        "[coreai] diffusion: %d prompt tok, %d generated, stop=%@, %d blocks, "
                        + "%d total steps, load=%.2fs gen=%.2fs\n",
                    result.promptTokenCount, result.generatedTokenCount, result.stopReason.rawValue,
                    result.blocks.count, result.totalSteps, result.modelLoadSeconds,
                    result.generateSeconds)
                FileHandle.standardError.write(Data(summary.utf8))
                for (i, b) in result.blocks.enumerated() {
                    let line = String(
                        format:
                            "[coreai]   block %d: %d steps, stop=%@, finalAccepted=%d, committed=%d tok, %.2fs\n",
                        i + 1, b.stepsRun, b.stopReason, b.finalAcceptedCount, b.committedTokens,
                        b.seconds)
                    FileHandle.standardError.write(Data(line.utf8))
                }
            } else if let draftPath {
                // Speculative decoding: target + draft pair.
                let result = try await CoreAIPipeline.runSpeculative(
                    targetPath: modelPath,
                    draftPath: draftPath,
                    prompt: promptText,
                    options: options,
                    draftTokens: draftTokens,
                    onToken: onToken)
                FileHandle.standardOutput.write(Data("\n".utf8))
                // The acceptance rate + speedup is the headline metric — always report it.
                let summary = String(
                    format:
                        "[coreai] speculative: %d prompt tok, %d generated, stop=%@, K=%d, "
                        + "%d/%d drafts accepted (%.1f%%), %.2f tok/target-pass, "
                        + "load=%.2fs prefill=%.2fs decode=%.2fs (%.1f tok/s)\n",
                    result.promptTokenCount, result.generatedTokenCount, result.stopReason.rawValue,
                    result.draftTokens, result.acceptedDraftTokens, result.draftedTokens,
                    result.acceptanceRate * 100, result.tokensPerTargetForward,
                    result.modelLoadSeconds, result.prefillSeconds, result.decodeSeconds,
                    result.decodeTokensPerSecond)
                FileHandle.standardError.write(Data(summary.utf8))
            } else {
                let result = try await CoreAIPipeline.run(
                    modelPath: modelPath,
                    prompt: promptText,
                    options: options,
                    onToken: onToken)
                // Newline after the streamed text, then a short summary to stderr.
                FileHandle.standardOutput.write(Data("\n".utf8))
                if verbose {
                    let summary = String(
                        format:
                            "[coreai] %d prompt tok, %d generated, stop=%@, load=%.2fs prefill=%.2fs decode=%.2fs (%.1f tok/s)\n",
                        result.promptTokenCount, result.generatedTokenCount, result.stopReason.rawValue,
                        result.modelLoadSeconds, result.prefillSeconds, result.decodeSeconds,
                        result.decodeTokensPerSecond)
                    FileHandle.standardError.write(Data(summary.utf8))
                }
            }
        } catch {
            FileHandle.standardError.write(Data("error: \(error)\n".utf8))
            exitCode = 1
        }
    }
    // Pump the main RunLoop while waiting so Metal/MPSGraph GPU-completion callbacks dispatched to
    // the main queue can fire. A bare `semaphore.wait()` blocks the main thread, which deadlocks
    // Apple's pipelined engine (custom Metal command queue + ComputeStream). The sequential engine
    // tolerates a blocked main thread, but the fast path does not.
    while semaphore.wait(timeout: .now()) == .timedOut {
        RunLoop.main.run(mode: .default, before: Date().addingTimeInterval(0.02))
    }
    exit(exitCode)
}

// MARK: - eagle subcommand

func eagleCommand(_ argv: [String]) {
    var target: String?
    var draft: String?
    var tokenizer: String?
    var prompt: String?
    var maxTokens = 64
    var draftTokens = 4   // Default for current EAGLE packages.
                          // K>=8 overflows the full-layer SDPA threadgroup limit on verify.
    var applyChatTemplate = true
    var verbose = false
    var targetOnly = false
    var draftUnrolled: String?
    var vocabSize = 262144
    var backbone = 2816
    var slidingWindow = 1024
    var maxContext = 4096

    func fail(_ m: String) -> Never {
        FileHandle.standardError.write(Data("error: \(m)\n".utf8)); exit(2)
    }
    var i = 0
    func value(_ f: String) -> String { i += 1; guard i < argv.count else { fail("missing value for \(f)") }; return argv[i] }
    func intValue(_ f: String) -> Int {
        guard let n = Int(value(f)) else { fail("invalid \(f)") }
        return n
    }
    while i < argv.count {
        switch argv[i] {
        case "--target": target = value(argv[i])
        case "--draft": draft = value(argv[i])
        case "--draft-unrolled": draftUnrolled = value(argv[i])
        case "--tokenizer": tokenizer = value(argv[i])
        case "--prompt", "-p": prompt = value(argv[i])
        case "--max-tokens", "-n": maxTokens = Int(value(argv[i])) ?? maxTokens
        case "--draft-tokens": draftTokens = Int(value(argv[i])) ?? draftTokens
        case "--raw", "--no-chat-template": applyChatTemplate = false
        case "--target-only": targetOnly = true
        case "--verbose", "-v": verbose = true
        case "--vocab", "--eagle-vocab": vocabSize = intValue(argv[i])
        case "--backbone", "--hidden-size", "--eagle-backbone", "--eagle-hidden-size":
            backbone = intValue(argv[i])
        case "--sliding-window", "--eagle-sliding-window": slidingWindow = intValue(argv[i])
        case "--max-context", "--eagle-max-context": maxContext = intValue(argv[i])
        default: fail("unknown eagle option: \(argv[i])")
        }
        i += 1
    }
    guard let targetPath = target else { fail("eagle requires --target <aimodel>") }
    guard let draftPath = draft else { fail("eagle requires --draft <aimodel>") }
    guard let tokDir = tokenizer else { fail("eagle requires --tokenizer <dir>") }
    guard let promptText = prompt else { fail("eagle requires --prompt") }

    let options = CoreAIPipeline.Options(
        maxTokens: maxTokens,
        temperature: 0,
        topK: nil,
        topP: nil,
        applyChatTemplate: applyChatTemplate,
        kvCapacity: nil,
        seed: nil,
        verbose: verbose)

    let targetP = targetPath, draftP = draftPath, tokD = tokDir, promptP = promptText
    let dt = draftTokens
    let tOnly = targetOnly
    let unrolledP = draftUnrolled
    let vocabP = vocabSize, backboneP = backbone, slidingP = slidingWindow, contextP = maxContext
    let semaphore = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var exitCode: Int32 = 0
    Task {
        defer { semaphore.signal() }
        do {
            let onToken: (String) -> Void = { FileHandle.standardOutput.write(Data($0.utf8)) }
            let r = try await CoreAIPipeline.runEagle(
                targetAimodel: targetP, draftAimodel: draftP, tokenizerDir: tokD,
                prompt: promptP, options: options, draftTokens: dt,
                vocabSize: vocabP, backbone: backboneP, slidingWindow: slidingP,
                maxContext: contextP, targetOnly: tOnly, draftUnrolledAimodel: unrolledP,
                onToken: onToken)
            FileHandle.standardOutput.write(Data("\n".utf8))
            let summary = String(
                format: "[coreai] eagle: %d prompt, %d generated, stop=%@, K=%d, "
                    + "%d/%d accepted (%.1f%%), %.2f tok/pass, load=%.2fs prefill=%.2fs "
                    + "decode=%.2fs (%.1f tok/s)\n",
                r.promptTokenCount, r.generatedTokenCount, r.stopReason.rawValue, r.draftTokens,
                r.acceptedDraftTokens, r.draftedTokens,
                r.draftedTokens > 0 ? Double(r.acceptedDraftTokens) / Double(r.draftedTokens) * 100 : 0,
                r.iterations > 0 ? Double(r.generatedTokenCount) / Double(r.iterations) : 0,
                r.modelLoadSeconds, r.prefillSeconds, r.decodeSeconds,
                r.decodeSeconds > 0 ? Double(r.generatedTokenCount) / r.decodeSeconds : 0)
            FileHandle.standardError.write(Data(summary.utf8))
        } catch {
            FileHandle.standardError.write(Data("eagle error: \(error)\n".utf8)); exitCode = 1
        }
    }
    while semaphore.wait(timeout: .now()) == .timedOut {
        RunLoop.main.run(mode: .default, before: Date().addingTimeInterval(0.02))
    }
    exit(exitCode)
}

// MARK: - bench subcommand

func benchCommand(_ argv: [String]) {
    var model: String?
    var seqs: [Int] = [1, 2, 4, 7, 1]
    var warmup = 4
    var iters = 10
    var i = 0
    while i < argv.count {
        switch argv[i] {
        case "--model", "-m":
            i += 1
            guard i < argv.count else { FileHandle.standardError.write(Data("error: --model needs a value\n".utf8)); exit(2) }
            model = argv[i]
        case "--seqs":
            i += 1
            guard i < argv.count else { break }
            seqs = argv[i].split(separator: ",").compactMap { Int($0) }
        case "--warmup":
            i += 1
            guard i < argv.count, let v = Int(argv[i]), v >= 0 else {
                FileHandle.standardError.write(Data("error: --warmup needs a non-negative integer\n".utf8)); exit(2)
            }
            warmup = v
        case "--iters":
            i += 1
            guard i < argv.count, let v = Int(argv[i]), v > 0 else {
                FileHandle.standardError.write(Data("error: --iters needs a positive integer\n".utf8)); exit(2)
            }
            iters = v
        case "-h", "--help":
            printUsage(); exit(0)
        default:
            FileHandle.standardError.write(Data("unknown bench option: \(argv[i])\n".utf8)); exit(2)
        }
        i += 1
    }
    guard let modelPath = model else {
        FileHandle.standardError.write(Data("error: bench requires --model <bundle>\n".utf8)); exit(2)
    }
    guard !seqs.isEmpty, seqs.allSatisfy({ $0 > 0 }) else {
        FileHandle.standardError.write(Data("error: --seqs must contain positive integers\n".utf8)); exit(2)
    }

    let semaphore = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var exitCode: Int32 = 0
    Task {
        defer { semaphore.signal() }
        do {
            _ = try await CoreAIPipeline.benchForward(
                modelPath: modelPath, seqLengths: seqs, warmup: warmup, iters: iters)
        } catch {
            FileHandle.standardError.write(Data("bench error: \(error)\n".utf8))
            exitCode = 1
        }
    }
    while semaphore.wait(timeout: .now()) == .timedOut {
        RunLoop.main.run(mode: .default, before: Date().addingTimeInterval(0.02))
    }
    exit(exitCode)
}

// MARK: - serve subcommand

func serveCommand(_ argv: [String]) {
    let cwd = FileManager.default.currentDirectoryPath
    var host = "127.0.0.1"
    var port = 1237
    var exportsDir = caixDefaultExportsPath()
    var registryPath = cwd + "/models/registry.json"
    var webDir = cwd + "/web"
    var convertScript = cwd + "/python/converter/convert.py"
    var python = "python3"
    var verbose = false
    var statsFile: String? = nil   // usage-stats persistence (default ~/.caix/usage.json)
    var prewarm = "smallest"
    var clusterManifest: String? = nil
    var clusterOptions = ClusterRuntimeOptions()
    // EAGLE MTP model. Enabled by default with the known bundle paths; disable with --no-eagle,
    // or override paths individually.
    var eagleEnabled = true
    var eagleName = "gemma-4-26b-a4b-mtp"
    var eagleTarget = cwd + "/exports/gemma-4-26b-a4b-eagle-target/eagle_target.aimodel"
    var eagleDraft = "/Volumes/SSD/ai-dev/eagle-resume/eagle_draft.aimodel"
    var eagleUnrolled: String? = "/Volumes/SSD/ai-dev/eagle-resume/eagle_draft_unrolled_k4.aimodel"
    var eagleTokenizer = cwd + "/exports/gemma-4-26b-a4b-coreai/tokenizer"
    var eagleVocab = 262144
    var eagleBackbone = 2816
    var eagleSlidingWindow = 1024
    var eagleMaxContext = 4096

    func fail(_ msg: String) -> Never {
        FileHandle.standardError.write(Data("error: \(msg)\n".utf8))
        exit(2)
    }

    var i = 0
    func value(_ flag: String) -> String {
        i += 1
        guard i < argv.count else { fail("missing value for \(flag)") }
        return argv[i]
    }
    func intValue(_ flag: String) -> Int {
        guard let n = Int(value(flag)) else { fail("invalid \(flag)") }
        return n
    }

    while i < argv.count {
        let arg = argv[i]
        switch arg {
        case "--port": guard let p = Int(value(arg)) else { fail("invalid --port") }; port = p
        case "--host": host = value(arg)
        case "--exports": exportsDir = value(arg)
        case "--registry": registryPath = value(arg)
        case "--web": webDir = value(arg)
        case "--convert-script": convertScript = value(arg)
        case "--python": python = value(arg)
        case "--stats-file": statsFile = value(arg)
        case "--prewarm": prewarm = value(arg)
        case "--no-prewarm": prewarm = "off"
        case "--cluster": clusterManifest = value(arg)
        case "--remote-stage": clusterOptions.remoteStageIDs.append(value(arg))
        case "--prompt-tokens":
            do {
                clusterOptions.promptTokens = try parseClusterTokenList(value(arg))
            } catch {
                fail("\(error)")
            }
        case "--max-tokens": clusterOptions.maxTokens = intValue(arg)
        case "--kv-capacity": clusterOptions.kvCapacity = intValue(arg)
        case "--max-context": clusterOptions.maxContextLength = intValue(arg)
        case "--join-timeout":
            do {
                clusterOptions.joinTimeoutSeconds = try parseClusterPositiveDouble(value(arg), flag: arg)
            } catch {
                fail("\(error)")
            }
        case "--once": clusterOptions.once = true
        case "--no-eagle": eagleEnabled = false
        case "--eagle-name": eagleName = value(arg)
        case "--eagle-target": eagleTarget = value(arg)
        case "--eagle-draft": eagleDraft = value(arg)
        case "--eagle-unrolled": eagleUnrolled = value(arg)
        case "--eagle-tokenizer": eagleTokenizer = value(arg)
        case "--eagle-vocab": eagleVocab = intValue(arg)
        case "--eagle-backbone", "--eagle-hidden-size": eagleBackbone = intValue(arg)
        case "--eagle-sliding-window": eagleSlidingWindow = intValue(arg)
        case "--eagle-max-context": eagleMaxContext = intValue(arg)
        case "--verbose", "-v":
            verbose = true
            clusterOptions.verbose = true
        case "-h", "--help": printUsage(); exit(0)
        default: fail("unknown option: \(arg)")
        }
        i += 1
    }

    if let clusterManifest {
        guard !clusterManifest.isEmpty else { fail("--cluster needs a manifest path") }
        let semaphore = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var exitCode: Int32 = 0
        Task {
            defer { semaphore.signal() }
            do {
                try await runClusterServeRuntime(
                    manifestPath: clusterManifest,
                    host: host,
                    port: port,
                    options: clusterOptions)
            } catch {
                FileHandle.standardError.write(Data("cluster serve error: \(error)\n".utf8))
                exitCode = 1
            }
        }
        while semaphore.wait(timeout: .now()) == .timedOut {
            RunLoop.main.run(mode: .default, before: Date().addingTimeInterval(0.02))
        }
        exit(exitCode)
    }

    // Build the EAGLE config only if the target+draft bundles actually exist on disk.
    var eagleConfig: EagleConfig? = nil
    if eagleEnabled {
        let fm = FileManager.default
        if fm.fileExists(atPath: eagleTarget) && fm.fileExists(atPath: eagleDraft) {
            let unrolled = eagleUnrolled.flatMap { fm.fileExists(atPath: $0) ? $0 : nil }
            eagleConfig = EagleConfig(
                name: eagleName, targetPath: eagleTarget, draftPath: eagleDraft,
                unrolledPath: unrolled, tokenizerDir: eagleTokenizer, vocab: eagleVocab,
                backbone: eagleBackbone, slidingWindow: eagleSlidingWindow,
                maxContext: eagleMaxContext)
            FileHandle.standardError.write(Data(
                "eagle MTP model: \(eagleName) (hidden: \(eagleBackbone), unrolled: \(unrolled != nil))\n".utf8))
        } else {
            FileHandle.standardError.write(Data(
                "note: EAGLE bundles not found (\(eagleTarget)); serving standard models only.\n".utf8))
        }
    }

    if !CoreAIPipeline.isLinked {
        FileHandle.standardError.write(
            Data(
                "note: Core AI runtime not linked — /v1 inference returns 503. Rebuild with COREAI_RUNTIME=1 for native generation.\n"
                    .utf8))
    }

    let semaphore = DispatchSemaphore(value: 0)
    nonisolated(unsafe) var exitCode: Int32 = 0
    Task {
        defer { semaphore.signal() }
        do {
            try await CoreAIServer.serve(
                host: host,
                port: port,
                exportsDir: exportsDir,
                registryPath: registryPath,
                webDir: webDir,
                convertScript: convertScript,
                pythonExecutable: python,
                caixVersion: CaixBuildInfo.version,
                verbose: verbose,
                eagleConfig: eagleConfig,
                statsFile: statsFile,
                prewarm: prewarm)
        } catch {
            FileHandle.standardError.write(Data("serve error: \(error)\n".utf8))
            exitCode = 1
        }
    }
    while semaphore.wait(timeout: .now()) == .timedOut {
        RunLoop.main.run(mode: .default, before: Date().addingTimeInterval(0.02))
    }
    exit(exitCode)
}
