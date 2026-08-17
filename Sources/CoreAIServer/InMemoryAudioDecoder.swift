import AudioToolbox
import Foundation

struct WhisperPCM: Sendable, Equatable {
    var samples: [Float]
    var sampleRate: Int
    var wasTruncated: Bool
}

/// Decodes an uploaded audio container directly from `Data` through AudioToolbox.
///
/// `AudioFileOpenWithCallbacks` lets CoreAudio sniff and decode the same common formats it handles
/// on disk without writing request data to a temporary path. `ExtAudioFile` performs native
/// channel mixing and sample-rate conversion into Whisper's fixed mono Float32/16 kHz contract.
enum InMemoryAudioDecoder {
    static let sampleRate = 16_000
    static let maximumSamples = sampleRate * 30
    private static let readChunkFrames: UInt32 = 8_192

    enum DecodingError: Error, Sendable, Equatable, CustomStringConvertible {
        case emptyInput
        case audioFileOpen(OSStatus)
        case extendedAudioFileOpen(OSStatus)
        case clientFormat(OSStatus)
        case read(OSStatus)
        case noAudioFrames

        var description: String {
            switch self {
            case .emptyInput: return "uploaded audio is empty"
            case .audioFileOpen(let status): return "AudioToolbox could not open audio (\(format(status)))"
            case .extendedAudioFileOpen(let status): return "AudioToolbox could not create a decoder (\(format(status)))"
            case .clientFormat(let status): return "AudioToolbox could not convert audio to 16 kHz mono (\(format(status)))"
            case .read(let status): return "AudioToolbox failed while decoding audio (\(format(status)))"
            case .noAudioFrames: return "uploaded file contains no decodable audio frames"
            }
        }

        private func format(_ status: OSStatus) -> String {
            let value = UInt32(bitPattern: status)
            let bytes: [UInt8] = [
                UInt8((value >> 24) & 0xff), UInt8((value >> 16) & 0xff),
                UInt8((value >> 8) & 0xff), UInt8(value & 0xff),
            ]
            if bytes.allSatisfy({ (0x20...0x7e).contains($0) }) {
                return "'\(String(bytes: bytes, encoding: .ascii) ?? "????")'"
            }
            return String(status)
        }
    }

    static func decodeToWhisperPCM(_ data: Data) throws -> WhisperPCM {
        guard !data.isEmpty else { throw DecodingError.emptyInput }
        let source = MemoryAudioSource(data: data)
        return try RetainedAudioCallbackContext.withPointer(to: source) { clientData in
            try decodeRetainedSource(clientData)
        }
    }

    private static func decodeRetainedSource(
        _ clientData: UnsafeMutableRawPointer
    ) throws -> WhisperPCM {
        var audioFile: AudioFileID?
        let openStatus = AudioFileOpenWithCallbacks(
            clientData,
            memoryAudioRead,
            nil,
            memoryAudioSize,
            nil,
            0,
            &audioFile)
        guard openStatus == noErr, let audioFile else {
            throw DecodingError.audioFileOpen(openStatus)
        }
        defer { AudioFileClose(audioFile) }

        var extendedFile: ExtAudioFileRef?
        let wrapStatus = ExtAudioFileWrapAudioFileID(audioFile, false, &extendedFile)
        guard wrapStatus == noErr, let extendedFile else {
            throw DecodingError.extendedAudioFileOpen(wrapStatus)
        }
        defer { ExtAudioFileDispose(extendedFile) }

        var clientFormat = AudioStreamBasicDescription(
            mSampleRate: Float64(sampleRate),
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagsNativeFloatPacked,
            mBytesPerPacket: UInt32(MemoryLayout<Float>.size),
            mFramesPerPacket: 1,
            mBytesPerFrame: UInt32(MemoryLayout<Float>.size),
            mChannelsPerFrame: 1,
            mBitsPerChannel: UInt32(MemoryLayout<Float>.size * 8),
            mReserved: 0)
        let formatStatus = withUnsafePointer(to: &clientFormat) { pointer in
            ExtAudioFileSetProperty(
                extendedFile,
                kExtAudioFileProperty_ClientDataFormat,
                UInt32(MemoryLayout<AudioStreamBasicDescription>.size),
                pointer)
        }
        guard formatStatus == noErr else { throw DecodingError.clientFormat(formatStatus) }

        var output: [Float] = []
        output.reserveCapacity(maximumSamples)
        while output.count < maximumSamples {
            let requested = UInt32(min(Int(readChunkFrames), maximumSamples - output.count))
            let chunk = try read(extendedFile, frameCount: requested)
            if chunk.isEmpty { break }
            output.append(contentsOf: chunk)
        }
        guard !output.isEmpty else { throw DecodingError.noAudioFrames }

        let wasTruncated: Bool
        if output.count == maximumSamples {
            wasTruncated = try !read(extendedFile, frameCount: 1).isEmpty
        } else {
            wasTruncated = false
        }
        return WhisperPCM(samples: output, sampleRate: sampleRate, wasTruncated: wasTruncated)
    }

    private static func read(_ file: ExtAudioFileRef, frameCount requested: UInt32) throws -> [Float] {
        var samples = [Float](repeating: 0, count: Int(requested))
        var frames = requested
        let status = samples.withUnsafeMutableBytes { bytes -> OSStatus in
            var buffers = AudioBufferList(
                mNumberBuffers: 1,
                mBuffers: AudioBuffer(
                    mNumberChannels: 1,
                    mDataByteSize: UInt32(bytes.count),
                    mData: bytes.baseAddress))
            return ExtAudioFileRead(file, &frames, &buffers)
        }
        guard status == noErr else { throw DecodingError.read(status) }
        if Int(frames) < samples.count { samples.removeLast(samples.count - Int(frames)) }
        return samples
    }
}

enum RetainedAudioCallbackContext {
    static func withPointer<Context: AnyObject, Result>(
        to context: Context,
        _ body: (UnsafeMutableRawPointer) throws -> Result
    ) rethrows -> Result {
        let retained = Unmanaged.passRetained(context)
        defer { retained.release() }
        return try body(retained.toOpaque())
    }
}

private final class MemoryAudioSource {
    let data: Data
    init(data: Data) { self.data = data }
}

private func memoryAudioRead(
    _ clientData: UnsafeMutableRawPointer,
    _ position: Int64,
    _ requestedCount: UInt32,
    _ destination: UnsafeMutableRawPointer,
    _ actualCount: UnsafeMutablePointer<UInt32>
) -> OSStatus {
    let source = Unmanaged<MemoryAudioSource>.fromOpaque(clientData).takeUnretainedValue()
    guard position >= 0, position < Int64(source.data.count) else {
        actualCount.pointee = 0
        return noErr
    }
    let start = Int(position)
    let count = min(Int(requestedCount), source.data.count - start)
    source.data.withUnsafeBytes { bytes in
        guard let base = bytes.baseAddress else { return }
        destination.copyMemory(from: base.advanced(by: start), byteCount: count)
    }
    actualCount.pointee = UInt32(count)
    return noErr
}

private func memoryAudioSize(_ clientData: UnsafeMutableRawPointer) -> Int64 {
    let source = Unmanaged<MemoryAudioSource>.fromOpaque(clientData).takeUnretainedValue()
    return Int64(source.data.count)
}
