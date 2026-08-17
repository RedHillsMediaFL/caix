import Foundation
import XCTest

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

@testable import PipelineRuntime

final class ResidentModelLockTests: XCTestCase {
    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private var canonicalLockURL: URL {
        repositoryRoot
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent(ResidentModelLock.filename)
    }

    private var gemma26A4BLockURL: URL {
        repositoryRoot
            .appendingPathComponent("models", isDirectory: true)
            .appendingPathComponent("gemma4-26b-a4b-whisper-lock.json")
    }

    func testCanonicalLockPinsApprovedResidentSourcesAndPrecisionSemantics() throws {
        let lock = try ResidentModelLock.load(from: canonicalLockURL)

        XCTAssertEqual(lock.schema, "caix.resident-model-lock.v1")
        XCTAssertEqual(lock.llm.publicModelID, "google/gemma-4-31B-it")
        XCTAssertEqual(
            lock.llm.target.repository,
            "google/gemma-4-31B-it-qat-q4_0-unquantized")
        XCTAssertEqual(
            lock.llm.target.revision,
            "1e4d8beecacb8b7590c1d8bedd7335f687bf311f")
        XCTAssertEqual(lock.llm.target.sourcePrecision, "bf16")
        XCTAssertEqual(lock.llm.target.qatRecipe, "q4_0")
        XCTAssertEqual(lock.llm.target.runtimePrecision, "q4_0")
        XCTAssertEqual(
            lock.llm.target.weights.map(\.path),
            ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
        XCTAssertEqual(lock.llm.target.weights.map(\.sizeBytes), [49_784_788_364, 12_761_549_884])

        XCTAssertEqual(
            lock.llm.assistant.repository,
            "google/gemma-4-31B-it-qat-q4_0-unquantized-assistant")
        XCTAssertEqual(
            lock.llm.assistant.revision,
            "96d4c8ca3cb38c107a8478587878124895d1e844")
        XCTAssertEqual(lock.llm.assistant.sourcePrecision, "bf16")
        XCTAssertEqual(lock.llm.assistant.qatRecipe, "q4_0")
        XCTAssertEqual(lock.llm.assistant.runtimePrecision, "fp16")
        XCTAssertEqual(lock.llm.assistant.weights.single?.path, "model.safetensors")
        XCTAssertEqual(lock.llm.assistant.weights.single?.sizeBytes, 939_042_560)

        XCTAssertEqual(
            lock.llm.chatTemplateSHA256,
            "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4")
        XCTAssertEqual(lock.llm.contextCandidates, [262_144, 131_072, 65_536, 32_768, 16_384])
        XCTAssertEqual(lock.llm.sampling.temperature, 1)
        XCTAssertEqual(lock.llm.sampling.topK, 64)
        XCTAssertEqual(lock.llm.sampling.topP, 0.95)
        XCTAssertEqual(lock.llm.sampling.eosTokenIDs, [1, 106, 50])
        XCTAssertEqual(lock.llm.sampling.padTokenID, 0)

        XCTAssertEqual(lock.speech.repository, "openai/whisper-large-v2")
        XCTAssertEqual(
            lock.speech.revision,
            "ae4642769ce2ad8fc292556ccea8e901f1530655")
        XCTAssertEqual(lock.speech.sourcePrecision, "fp32")
        XCTAssertEqual(lock.speech.runtimePrecision, "fp16")
        XCTAssertEqual(lock.speech.weights.single?.sizeBytes, 6_173_370_152)
        XCTAssertEqual(
            lock.speech.metadata.addedTokensSHA256,
            "9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1")
        XCTAssertEqual(
            lock.speech.metadata.mergesSHA256,
            "2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6")
        XCTAssertEqual(
            lock.speech.metadata.vocabularySHA256,
            "8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9")
        XCTAssertEqual(
            lock.speech.metadata.normalizerSHA256,
            "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd")
        XCTAssertEqual(lock.speech.geometry.sampleRateHz, 16_000)
        XCTAssertEqual(lock.speech.geometry.melBins, 80)
        XCTAssertEqual(lock.speech.geometry.windowSamples, 480_000)
    }

    func testGemma26A4BLockPinsMatchedQATQ4SourcesAndCurrentChatTemplate() throws {
        let lock = try ResidentModelLock.load(from: gemma26A4BLockURL)

        XCTAssertEqual(lock.schema, "caix.resident-model-lock.v1")
        XCTAssertEqual(lock.deployment, "gemma4-26b-a4b-whisper-large-v2")
        XCTAssertEqual(lock.llm.publicModelID, "google/gemma-4-26B-A4B-it")
        XCTAssertEqual(
            lock.llm.target.repository,
            "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized")
        XCTAssertEqual(
            lock.llm.target.revision,
            "f1e06dc520982d9b9edd76859fdb7ab209449949")
        XCTAssertEqual(lock.llm.target.sourcePrecision, "bf16")
        XCTAssertEqual(lock.llm.target.qatRecipe, "q4_0")
        XCTAssertEqual(lock.llm.target.runtimePrecision, "q4_0")
        XCTAssertEqual(
            lock.llm.target.weights.map(\.sizeBytes),
            [49_907_246_508, 1_704_763_408])

        XCTAssertEqual(
            lock.llm.assistant.repository,
            "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant")
        XCTAssertEqual(
            lock.llm.assistant.revision,
            "9537141506fe8875b3ed45b264af13580cb29166")
        XCTAssertEqual(lock.llm.assistant.sourcePrecision, "bf16")
        XCTAssertEqual(lock.llm.assistant.qatRecipe, "q4_0")
        XCTAssertEqual(lock.llm.assistant.runtimePrecision, "q4_0")
        XCTAssertEqual(lock.llm.assistant.weights.single?.sizeBytes, 839_427_840)

        XCTAssertEqual(
            lock.llm.chatTemplateSHA256,
            "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4")
        XCTAssertEqual(lock.llm.target.metadata.chatTemplateSHA256, lock.llm.chatTemplateSHA256)
        XCTAssertEqual(lock.llm.assistant.metadata.chatTemplateSHA256, lock.llm.chatTemplateSHA256)
        XCTAssertEqual(lock.llm.geometry.hiddenSize, 2_816)
        XCTAssertEqual(lock.llm.geometry.layers, 30)
        XCTAssertEqual(lock.llm.geometry.attentionHeads, 16)
        XCTAssertEqual(lock.llm.geometry.slidingKVHeads, 8)
        XCTAssertEqual(lock.llm.geometry.globalKVHeads, 2)
        XCTAssertEqual(lock.llm.geometry.slidingLayers, 25)
        XCTAssertEqual(lock.llm.geometry.fullLayers, 5)
        XCTAssertEqual(lock.llm.assistantGeometry.backboneHiddenSize, 2_816)
        XCTAssertEqual(lock.llm.assistantGeometry.attentionHeads, 16)
        XCTAssertEqual(lock.llm.assistantGeometry.kvHeads, 8)
        XCTAssertEqual(lock.llm.assistantGeometry.globalKVHeads, 2)

        let existingLock = try ResidentModelLock.load(from: canonicalLockURL)
        XCTAssertEqual(lock.speech, existingLock.speech, "Whisper contract must remain unchanged")
    }

    func testGemma26A4BValidationRejectsAssistantFrom31BContract() throws {
        let url = try mutatedLock(
            at: gemma26A4BLockURL,
            replacing: "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant",
            with: "google/gemma-4-31B-it-qat-q4_0-unquantized-assistant")

        assertContractViolation("llm.assistant.repository") {
            _ = try ResidentModelLock.load(from: url)
        }
    }

    func testGemma26A4BValidationRequiresQ4AssistantRuntimePrecision() throws {
        let url = try mutatedLock(
            at: gemma26A4BLockURL,
            replacing: """
            "runtime_precision": "q4_0",
                  "weights": [
                    {
                      "path": "model.safetensors"
            """,
            with: """
            "runtime_precision": "fp16",
                  "weights": [
                    {
                      "path": "model.safetensors"
            """,
            expectedOccurrences: 1)

        assertContractViolation("llm.assistant.runtime_precision") {
            _ = try ResidentModelLock.load(from: url)
        }
    }

    func testValidationRejectsUnapprovedTargetRevision() throws {
        let url = try mutatedLock(
            replacing: "1e4d8beecacb8b7590c1d8bedd7335f687bf311f",
            with: "0000000000000000000000000000000000000000")

        assertContractViolation("llm.target.revision") {
            _ = try ResidentModelLock.load(from: url)
        }
    }

    func testValidationNeverLabelsUnquantizedQATSourceAsFourBit() throws {
        let url = try mutatedLock(
            replacing: #""source_precision": "bf16""#,
            with: #""source_precision": "q4_0""#,
            expectedOccurrences: 2)

        assertContractViolation("llm.target.source_precision") {
            _ = try ResidentModelLock.load(from: url)
        }
    }

    func testValidationRequiresLowercaseSHA256() throws {
        let digest = "95b105ad23b315067985415e721ab9c19cfcf90918b34ea0fa479a08489d86b7"
        let url = try mutatedLock(replacing: digest, with: digest.uppercased())

        assertContractViolation("llm.target.metadata.config_sha256") {
            _ = try ResidentModelLock.load(from: url)
        }
    }

    func testValidationRequiresExactWeightIdentityAndSize() throws {
        let url = try mutatedLock(replacing: "49784788364", with: "49784788363")

        assertContractViolation("llm.target.weights") {
            _ = try ResidentModelLock.load(from: url)
        }
    }

    func testValidationRequiresExactGemmaGeometryAndContextLadder() throws {
        let geometryURL = try mutatedLock(
            replacing: #""hidden_size": 5376"#,
            with: #""hidden_size": 5375"#)
        assertContractViolation("llm.geometry") {
            _ = try ResidentModelLock.load(from: geometryURL)
        }

        let contextURL = try mutatedLock(
            replacing: "262144, 131072, 65536, 32768, 16384",
            with: "262144, 131072, 65536, 32768, 8192")
        assertContractViolation("llm.context_candidates") {
            _ = try ResidentModelLock.load(from: contextURL)
        }
    }

    func testValidationRequiresWhisperLargeV2FrontendAndGeometry() throws {
        let url = try mutatedLock(
            replacing: #""mel_bins": 80"#,
            with: #""mel_bins": 128"#)

        assertContractViolation("speech.geometry") {
            _ = try ResidentModelLock.load(from: url)
        }
    }

    func testValidationPinsEveryWhisperTokenizerAndNormalizationAsset() throws {
        let cases = [
            (
                "9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1",
                "speech.metadata.added_tokens_sha256"),
            (
                "2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6",
                "speech.metadata.merges_sha256"),
            (
                "8f680bba319e01a653d2e8a5dbc17a9157179e0576e6ce74ce0c06356c6e24f9",
                "speech.metadata.vocab_sha256"),
            (
                "bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd",
                "speech.metadata.normalizer_sha256"),
        ]

        for (digest, field) in cases {
            let url = try mutatedLock(
                replacing: digest,
                with: String(repeating: "0", count: 64))
            assertContractViolation(field) {
                _ = try ResidentModelLock.load(from: url)
            }
        }
    }

    func testLoaderRejectsUnsupportedSchema() throws {
        let url = try mutatedLock(
            replacing: "caix.resident-model-lock.v1",
            with: "caix.resident-model-lock.v2")

        XCTAssertThrowsError(try ResidentModelLock.load(from: url)) { error in
            XCTAssertEqual(
                error as? ResidentModelLockError,
                .unsupportedSchema("caix.resident-model-lock.v2"))
        }
    }

    func testLoaderRejectsSymlinkAndNonRegularFile() throws {
        let parent = temporaryDirectory()
        let symlink = parent.appendingPathComponent("lock-link.json")
        try FileManager.default.createSymbolicLink(at: symlink, withDestinationURL: canonicalLockURL)

        XCTAssertThrowsError(try ResidentModelLock.load(from: symlink)) { error in
            XCTAssertEqual(error as? ResidentModelLockError, .notRegularFile)
        }
        XCTAssertThrowsError(try ResidentModelLock.load(from: parent)) { error in
            XCTAssertEqual(error as? ResidentModelLockError, .notRegularFile)
        }
    }

    func testLoaderRejectsOversizedLockBeforeDecode() throws {
        let url = temporaryDirectory().appendingPathComponent("oversized.json")
        try Data(repeating: 0x20, count: ResidentModelLock.maximumLockBytes + 1).write(to: url)

        XCTAssertThrowsError(try ResidentModelLock.load(from: url)) { error in
            XCTAssertEqual(error as? ResidentModelLockError, .fileTooLarge)
        }
    }

    func testDescriptorFirstReaderAcceptsRegularFileAndRejectsSymlinkAndFIFO() throws {
        let parent = temporaryDirectory()
        let regular = parent.appendingPathComponent("regular.json")
        let symlink = parent.appendingPathComponent("regular-link.json")
        let fifo = parent.appendingPathComponent("lock.fifo")
        let expected = Data(#"{"ready":true}"#.utf8)
        try expected.write(to: regular)
        try FileManager.default.createSymbolicLink(at: symlink, withDestinationURL: regular)

        XCTAssertEqual(
            try BoundedRegularFileReader.read(regular, maximumBytes: expected.count),
            expected)
        XCTAssertThrowsError(
            try BoundedRegularFileReader.read(symlink, maximumBytes: expected.count))

        #if canImport(Darwin)
        XCTAssertEqual(Darwin.mkfifo(fifo.path, mode_t(0o600)), 0)
        let keeper = Darwin.open(fifo.path, O_RDWR | O_NONBLOCK | O_CLOEXEC)
        #elseif canImport(Glibc)
        XCTAssertEqual(Glibc.mkfifo(fifo.path, mode_t(0o600)), 0)
        let keeper = Glibc.open(fifo.path, O_RDWR | O_NONBLOCK | O_CLOEXEC)
        #else
        throw XCTSkip("POSIX FIFO behavior is unavailable on this platform")
        #endif
        XCTAssertGreaterThanOrEqual(keeper, 0)
        defer {
            #if canImport(Darwin)
            Darwin.close(keeper)
            #elseif canImport(Glibc)
            Glibc.close(keeper)
            #endif
        }
        XCTAssertThrowsError(
            try BoundedRegularFileReader.read(fifo, maximumBytes: expected.count))
    }

    private func assertContractViolation(
        _ expectedField: String,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ body: () throws -> Void
    ) {
        XCTAssertThrowsError(try body(), file: file, line: line) { error in
            XCTAssertEqual(
                error as? ResidentModelLockError,
                .contractViolation(expectedField),
                file: file,
                line: line)
        }
    }

    private func mutatedLock(
        replacing old: String,
        with new: String,
        expectedOccurrences: Int = 1
    ) throws -> URL {
        try mutatedLock(
            at: canonicalLockURL,
            replacing: old,
            with: new,
            expectedOccurrences: expectedOccurrences)
    }

    private func mutatedLock(
        at sourceURL: URL,
        replacing old: String,
        with new: String,
        expectedOccurrences: Int = 1
    ) throws -> URL {
        var contents = try String(contentsOf: sourceURL, encoding: .utf8)
        let occurrences = contents.components(separatedBy: old).count - 1
        XCTAssertEqual(occurrences, expectedOccurrences, "fixture mutation must be unambiguous")
        contents = contents.replacingOccurrences(of: old, with: new)
        let url = temporaryDirectory().appendingPathComponent("mutated-lock.json")
        try Data(contents.utf8).write(to: url)
        return url
    }

    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("caix-resident-lock-tests-\(UUID().uuidString)", isDirectory: true)
        try! FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
}

private extension Array {
    var single: Element? { count == 1 ? first : nil }
}
