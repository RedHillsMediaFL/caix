import Accelerate
import Foundation

/// Native implementation of the exact Whisper-large-v2 feature-extractor contract.
///
/// The 400-point DFT is deliberately evaluated as two Accelerate SGEMMs because Accelerate's
/// real DFT setup does not implement length 400. This still batches every one of the 3,000 frames
/// into optimized matrix operations and is much faster than issuing 603,000 scalar dot products.
enum WhisperLogMelExtractor {
    static let sampleRate = 16_000
    static let maximumSamples = sampleRate * 30
    static let melBinCount = 80
    static let frameCount = 3_000
    private static let fftSize = 400
    private static let frequencyBinCount = fftSize / 2 + 1
    private static let hopLength = 160

    enum ExtractionError: Error, Sendable, Equatable, CustomStringConvertible {
        case tooManySamples
        case matrixFailure

        var description: String {
            switch self {
            case .tooManySamples: return "audio exceeds Whisper's 30-second feature window"
            case .matrixFailure: return "failed to construct Whisper log-mel features"
            }
        }
    }

    static func extract(samples: [Float]) throws -> [Float] {
        guard samples.count <= maximumSamples else { throw ExtractionError.tooManySamples }
        guard samples.allSatisfy(\.isFinite) else { throw ExtractionError.matrixFailure }
        if samples.allSatisfy({ $0 == 0 }) {
            return [Float](repeating: -1.5, count: melBinCount * frameCount)
        }

        var padded = [Float](repeating: 0, count: maximumSamples)
        padded.replaceSubrange(0..<samples.count, with: samples)

        // Row-major [frame, sample]. torch.stft(center=True, pad_mode="reflect") places frame
        // centers at frame*hop and reflects 200 samples on each edge.
        var framed = [Float](repeating: 0, count: frameCount * fftSize)
        for frame in 0..<frameCount {
            let center = frame * hopLength
            let base = frame * fftSize
            for sample in 0..<fftSize {
                let sourceIndex = reflectedIndex(center + sample - fftSize / 2)
                framed[base + sample] = padded[sourceIndex] * hannWindow[sample]
            }
        }

        let real = matrixMultiply(
            framed, rows: frameCount, inner: fftSize,
            dftCosine, columns: frequencyBinCount)
        let imaginary = matrixMultiply(
            framed, rows: frameCount, inner: fftSize,
            dftSine, columns: frequencyBinCount)
        guard real.count == frameCount * frequencyBinCount,
              imaginary.count == real.count
        else { throw ExtractionError.matrixFailure }

        var power = [Float](repeating: 0, count: real.count)
        for index in power.indices {
            power[index] = real[index] * real[index] + imaginary[index] * imaginary[index]
        }
        var frameMajorMels = matrixMultiply(
            power, rows: frameCount, inner: frequencyBinCount,
            melFilters, columns: melBinCount)
        guard frameMajorMels.count == frameCount * melBinCount else {
            throw ExtractionError.matrixFailure
        }

        var maximumLog = -Float.infinity
        for index in frameMajorMels.indices {
            let value = log10f(max(frameMajorMels[index], 1e-10))
            frameMajorMels[index] = value
            maximumLog = max(maximumLog, value)
        }
        let floor = maximumLog - 8
        var melMajor = [Float](repeating: 0, count: melBinCount * frameCount)
        for frame in 0..<frameCount {
            for mel in 0..<melBinCount {
                let value = max(frameMajorMels[frame * melBinCount + mel], floor)
                melMajor[mel * frameCount + frame] = (value + 4) / 4
            }
        }
        return melMajor
    }

    private static func reflectedIndex(_ rawIndex: Int) -> Int {
        var index = rawIndex
        while index < 0 || index >= maximumSamples {
            if index < 0 {
                index = -index
            } else {
                index = 2 * maximumSamples - 2 - index
            }
        }
        return index
    }

    private static func matrixMultiply(
        _ left: [Float], rows: Int, inner: Int,
        _ right: [Float], columns: Int
    ) -> [Float] {
        guard left.count == rows * inner, right.count == inner * columns,
              rows <= Int(Int32.max), inner <= Int(Int32.max), columns <= Int(Int32.max)
        else { return [] }
        var output = [Float](repeating: 0, count: rows * columns)
        left.withUnsafeBufferPointer { leftBuffer in
            right.withUnsafeBufferPointer { rightBuffer in
                output.withUnsafeMutableBufferPointer { outputBuffer in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Int32(rows), Int32(columns), Int32(inner),
                        1,
                        leftBuffer.baseAddress, Int32(inner),
                        rightBuffer.baseAddress, Int32(columns),
                        0,
                        outputBuffer.baseAddress, Int32(columns))
                }
            }
        }
        return output
    }

    private static let hannWindow: [Float] = (0..<fftSize).map { index in
        Float(0.5 - 0.5 * cos(2 * Double.pi * Double(index) / Double(fftSize)))
    }

    /// Row-major [time sample, frequency bin]. The sine sign is irrelevant after magnitude².
    private static let dftCosine: [Float] = {
        var matrix = [Float](repeating: 0, count: fftSize * frequencyBinCount)
        for sample in 0..<fftSize {
            for frequency in 0..<frequencyBinCount {
                let phase = 2 * Double.pi * Double(sample * frequency) / Double(fftSize)
                matrix[sample * frequencyBinCount + frequency] = Float(cos(phase))
            }
        }
        return matrix
    }()

    private static let dftSine: [Float] = {
        var matrix = [Float](repeating: 0, count: fftSize * frequencyBinCount)
        for sample in 0..<fftSize {
            for frequency in 0..<frequencyBinCount {
                let phase = 2 * Double.pi * Double(sample * frequency) / Double(fftSize)
                matrix[sample * frequencyBinCount + frequency] = Float(sin(phase))
            }
        }
        return matrix
    }()

    /// Row-major [frequency bin, mel bin], equivalent to Transformers 5.6.2's
    /// `mel_filter_bank(..., norm="slaney", mel_scale="slaney")`.
    private static let melFilters: [Float] = {
        let minimumMel = hertzToSlaneyMel(0)
        let maximumMel = hertzToSlaneyMel(Double(sampleRate) / 2)
        let filterFrequencies = (0..<(melBinCount + 2)).map { index in
            slaneyMelToHertz(
                minimumMel
                    + (maximumMel - minimumMel) * Double(index) / Double(melBinCount + 1))
        }
        var filters = [Float](repeating: 0, count: frequencyBinCount * melBinCount)
        for frequencyBin in 0..<frequencyBinCount {
            let frequency = Double(frequencyBin) * Double(sampleRate) / Double(fftSize)
            for mel in 0..<melBinCount {
                let lower = filterFrequencies[mel]
                let center = filterFrequencies[mel + 1]
                let upper = filterFrequencies[mel + 2]
                let down = (frequency - lower) / (center - lower)
                let up = (upper - frequency) / (upper - center)
                let triangle = max(0, min(down, up))
                let slaneyNormalization = 2 / (upper - lower)
                filters[frequencyBin * melBinCount + mel] =
                    Float(triangle * slaneyNormalization)
            }
        }
        return filters
    }()

    private static func hertzToSlaneyMel(_ frequency: Double) -> Double {
        if frequency < 1_000 { return 3 * frequency / 200 }
        return 15 + log(frequency / 1_000) * 27 / log(6.4)
    }

    private static func slaneyMelToHertz(_ mel: Double) -> Double {
        if mel < 15 { return 200 * mel / 3 }
        return 1_000 * exp(log(6.4) / 27 * (mel - 15))
    }
}
