#if COREAI_RUNTIME

import Foundation

/// Hot server handle for the native Qwen3.8 target plus its MTP sidecar.
public final class PersistentQwen38MTPModel: @unchecked Sendable {
    private let engine: Qwen38NativeMTPEngine
    private let proof: Qwen38MTPProof?
    public let name: String
    public let bundleByteSize: UInt64

    public static func load(bundlePath: String, verbose: Bool = false) async throws
        -> PersistentQwen38MTPModel
    {
        let bundle = try ResolvedBundle.load(at: bundlePath)
        guard let mtp = bundle.qwen38?.mtp, bundle.mtpAimodelURL != nil else {
            throw CoreAIPipeline.RuntimeError.invalidBundle(
                "persistent Qwen3.8 MTP requires a declared sidecar")
        }
        let engine = try await Qwen38NativeMTPEngine.load(bundle: bundle, verbose: verbose)
        return PersistentQwen38MTPModel(
            engine: engine,
            proof: mtp.proof,
            name: bundle.name,
            bundleByteSize: Self.directorySize(bundle.root))
    }

    private init(
        engine: Qwen38NativeMTPEngine,
        proof: Qwen38MTPProof?,
        name: String,
        bundleByteSize: UInt64
    ) {
        self.engine = engine
        self.proof = proof
        self.name = name
        self.bundleByteSize = bundleByteSize
    }

    public func generate(
        messages: [[String: String]],
        options: CoreAIPipeline.Options,
        tools: [[String: any Sendable]]? = nil,
        onToken: ((String) -> Void)? = nil
    ) async throws -> CoreAIPipeline.Result {
        let tokens = try engine.encodePrompt(
            messages: messages,
            tools: tools,
            applyChatTemplate: options.applyChatTemplate)
        let decision: Qwen38AccelerationDecision
        do {
            decision = try Qwen38ExecutionPolicy.resolve(
                requested: options.acceleration,
                temperature: options.temperature,
                proof: proof,
                nativeMTPAvailable: true)
        } catch {
            throw CoreAIPipeline.RuntimeError.unsupportedFeature(
                "native Qwen3.8 MTP is unavailable: \(error)")
        }
        if decision == .nativeMTP {
            return try await engine.generate(
                promptTokens: tokens, options: options, onToken: onToken)
        }
        return try await engine.generateAutoregressive(
            promptTokens: tokens, options: options, onToken: onToken)
    }

    private static func directorySize(_ root: URL) -> UInt64 {
        let keys: [URLResourceKey] = [.isRegularFileKey, .fileAllocatedSizeKey]
        guard let files = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: keys)
        else { return 0 }
        var total: UInt64 = 0
        for case let url as URL in files {
            guard let values = try? url.resourceValues(forKeys: Set(keys)),
                values.isRegularFile == true
            else { continue }
            total += UInt64(values.fileAllocatedSize ?? 0)
        }
        return total
    }
}

#endif
