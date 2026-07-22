import Foundation

#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

enum BoundedRegularFileReaderError: Error, Equatable, Sendable {
    case unsupportedPlatform
    case invalidLimit
    case cannotOpen(Int32)
    case cannotInspect(Int32)
    case notRegularFile
    case invalidSize
    case emptyFile
    case tooLarge
    case cannotRead(Int32)
    case changedWhileReading
}

/// Descriptor-first bounded reader for small control-plane files.
///
/// The final path component is opened with `O_NOFOLLOW` before its type or size is trusted. The
/// opened descriptor is then inspected with `fstat`, so a path replacement cannot redirect the
/// subsequent read. `O_NONBLOCK` prevents a malicious FIFO from hanging the process before the
/// regular-file check; it has no effect on regular-file reads.
enum BoundedRegularFileReader {
    private static let chunkBytes = 64 * 1_024

    static func read(
        _ url: URL,
        maximumBytes: Int,
        minimumBytes: Int = 1,
        exactBytes: Int? = nil
    ) throws -> Data {
        guard maximumBytes >= 0,
              minimumBytes >= 0,
              minimumBytes <= maximumBytes,
              exactBytes.map({ $0 >= minimumBytes && $0 <= maximumBytes }) ?? true
        else { throw BoundedRegularFileReaderError.invalidLimit }

        let descriptor = try openDescriptor(url.path)
        defer { closeDescriptor(descriptor) }

        let initialSize = try regularFileSize(descriptor)
        guard initialSize >= UInt64(minimumBytes) else {
            throw BoundedRegularFileReaderError.emptyFile
        }
        guard initialSize <= UInt64(maximumBytes) else {
            throw BoundedRegularFileReaderError.tooLarge
        }
        if let exactBytes, initialSize != UInt64(exactBytes) {
            throw BoundedRegularFileReaderError.invalidSize
        }
        guard let capacity = Int(exactly: initialSize) else {
            throw BoundedRegularFileReaderError.invalidSize
        }

        var data = Data()
        data.reserveCapacity(capacity)
        var buffer = [UInt8](
            repeating: 0,
            count: max(1, min(chunkBytes, maximumBytes)))
        var total: UInt64 = 0
        while true {
            let count = try readChunk(descriptor, into: &buffer)
            if count == 0 { break }
            let (newTotal, overflow) = total.addingReportingOverflow(UInt64(count))
            guard !overflow,
                  newTotal <= initialSize,
                  newTotal <= UInt64(maximumBytes)
            else { throw BoundedRegularFileReaderError.changedWhileReading }
            data.append(contentsOf: buffer[0..<count])
            total = newTotal
        }

        let finalSize = try regularFileSize(descriptor)
        guard total == initialSize, finalSize == initialSize else {
            throw BoundedRegularFileReaderError.changedWhileReading
        }
        return data
    }

    private static func openDescriptor(_ path: String) throws -> Int32 {
        #if canImport(Darwin)
        let descriptor = path.withCString {
            Darwin.open($0, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
        }
        guard descriptor >= 0 else {
            let code = errno
            if code == ELOOP || code == ENOENT || code == ENOTDIR {
                throw BoundedRegularFileReaderError.notRegularFile
            }
            throw BoundedRegularFileReaderError.cannotOpen(code)
        }
        return descriptor
        #elseif canImport(Glibc)
        let descriptor = path.withCString {
            Glibc.open($0, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
        }
        guard descriptor >= 0 else {
            let code = errno
            if code == ELOOP || code == ENOENT || code == ENOTDIR {
                throw BoundedRegularFileReaderError.notRegularFile
            }
            throw BoundedRegularFileReaderError.cannotOpen(code)
        }
        return descriptor
        #else
        throw BoundedRegularFileReaderError.unsupportedPlatform
        #endif
    }

    private static func regularFileSize(_ descriptor: Int32) throws -> UInt64 {
        #if canImport(Darwin)
        var info = stat()
        guard Darwin.fstat(descriptor, &info) == 0 else {
            throw BoundedRegularFileReaderError.cannotInspect(errno)
        }
        guard (info.st_mode & mode_t(S_IFMT)) == mode_t(S_IFREG) else {
            throw BoundedRegularFileReaderError.notRegularFile
        }
        guard info.st_size >= 0, let size = UInt64(exactly: info.st_size) else {
            throw BoundedRegularFileReaderError.invalidSize
        }
        return size
        #elseif canImport(Glibc)
        var info = stat()
        guard Glibc.fstat(descriptor, &info) == 0 else {
            throw BoundedRegularFileReaderError.cannotInspect(errno)
        }
        guard (info.st_mode & mode_t(S_IFMT)) == mode_t(S_IFREG) else {
            throw BoundedRegularFileReaderError.notRegularFile
        }
        guard info.st_size >= 0, let size = UInt64(exactly: info.st_size) else {
            throw BoundedRegularFileReaderError.invalidSize
        }
        return size
        #else
        throw BoundedRegularFileReaderError.unsupportedPlatform
        #endif
    }

    private static func readChunk(
        _ descriptor: Int32,
        into buffer: inout [UInt8]
    ) throws -> Int {
        while true {
            #if canImport(Darwin)
            let count = buffer.withUnsafeMutableBytes { rawBuffer in
                Darwin.read(descriptor, rawBuffer.baseAddress, rawBuffer.count)
            }
            #elseif canImport(Glibc)
            let count = buffer.withUnsafeMutableBytes { rawBuffer in
                Glibc.read(descriptor, rawBuffer.baseAddress, rawBuffer.count)
            }
            #else
            throw BoundedRegularFileReaderError.unsupportedPlatform
            #endif
            if count >= 0 { return count }
            #if canImport(Darwin) || canImport(Glibc)
            if errno == EINTR { continue }
            throw BoundedRegularFileReaderError.cannotRead(errno)
            #endif
        }
    }

    private static func closeDescriptor(_ descriptor: Int32) {
        #if canImport(Darwin)
        _ = Darwin.close(descriptor)
        #elseif canImport(Glibc)
        _ = Glibc.close(descriptor)
        #endif
    }
}
