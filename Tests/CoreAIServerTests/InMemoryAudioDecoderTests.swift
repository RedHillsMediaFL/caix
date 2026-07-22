import XCTest

@testable import CoreAIServer

final class InMemoryAudioDecoderTests: XCTestCase {
    func testDecodesAndResamplesPCM16WAVFromMemory() throws {
        let sourceRate = 8_000
        let frames = sourceRate
        let samples = (0..<frames).map { index in
            Float(sin(2 * Double.pi * 440 * Double(index) / Double(sourceRate)) * 0.5)
        }
        let wav = makePCM16WAV(samples: samples, sampleRate: sourceRate, channels: 1)

        let decoded = try InMemoryAudioDecoder.decodeToWhisperPCM(wav)

        XCTAssertEqual(decoded.sampleRate, 16_000)
        XCTAssertEqual(decoded.samples.count, 16_000, accuracy: 4)
        XCTAssertTrue(decoded.samples.allSatisfy(\.isFinite))
        XCTAssertLessThanOrEqual(decoded.samples.map(abs).max() ?? 0, 0.55)
        XCTAssertGreaterThan(decoded.samples.map(abs).max() ?? 0, 0.4)
    }

    func testDownmixesStereoAndCapsAtThirtySeconds() throws {
        let sampleRate = 16_000
        let frameCount = sampleRate * 31
        var interleaved: [Float] = []
        interleaved.reserveCapacity(frameCount * 2)
        for index in 0..<frameCount {
            let sample = Float(sin(2 * Double.pi * 220 * Double(index) / Double(sampleRate)) * 0.25)
            interleaved.append(sample)
            interleaved.append(sample)
        }
        let wav = makePCM16WAV(
            samples: interleaved, sampleRate: sampleRate, channels: 2)

        let decoded = try InMemoryAudioDecoder.decodeToWhisperPCM(wav)

        XCTAssertEqual(decoded.samples.count, InMemoryAudioDecoder.maximumSamples)
        XCTAssertTrue(decoded.wasTruncated)
        XCTAssertGreaterThan(decoded.samples.map(abs).max() ?? 0, 0.1)
    }

    func testRejectsInvalidAndEmptyAudio() {
        XCTAssertThrowsError(try InMemoryAudioDecoder.decodeToWhisperPCM(Data()))
        XCTAssertThrowsError(
            try InMemoryAudioDecoder.decodeToWhisperPCM(Data("not audio".utf8)))
    }

    private func makePCM16WAV(
        samples: [Float], sampleRate: Int, channels: Int
    ) -> Data {
        precondition(samples.count % channels == 0)
        var pcm = Data()
        pcm.reserveCapacity(samples.count * 2)
        for sample in samples {
            let scaled = Int16((max(-1, min(1, sample)) * Float(Int16.max)).rounded())
            appendLE(UInt16(bitPattern: scaled), to: &pcm)
        }

        let byteRate = UInt32(sampleRate * channels * 2)
        let blockAlign = UInt16(channels * 2)
        var wav = Data("RIFF".utf8)
        appendLE(UInt32(36 + pcm.count), to: &wav)
        wav.append(Data("WAVEfmt ".utf8))
        appendLE(UInt32(16), to: &wav)
        appendLE(UInt16(1), to: &wav)
        appendLE(UInt16(channels), to: &wav)
        appendLE(UInt32(sampleRate), to: &wav)
        appendLE(byteRate, to: &wav)
        appendLE(blockAlign, to: &wav)
        appendLE(UInt16(16), to: &wav)
        wav.append(Data("data".utf8))
        appendLE(UInt32(pcm.count), to: &wav)
        wav.append(pcm)
        return wav
    }

    private func appendLE<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var littleEndian = value.littleEndian
        withUnsafeBytes(of: &littleEndian) { data.append(contentsOf: $0) }
    }
}
