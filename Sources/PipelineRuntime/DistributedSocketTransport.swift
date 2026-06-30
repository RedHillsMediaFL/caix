import Darwin
import Foundation

public enum DistributedSocketTransportError: Error, Equatable, Sendable, CustomStringConvertible {
    case badAddress(String)
    case socketError(operation: String, errno: Int32)
    case connectionClosed

    public var description: String {
        switch self {
        case .badAddress(let value):
            return "Bad socket address: \(value)"
        case .socketError(let operation, let code):
            return "\(operation) failed: \(String(cString: strerror(code)))"
        case .connectionClosed:
            return "Socket connection closed"
        }
    }
}

public final class DistributedSocketWorkerListener: @unchecked Sendable {
    private let fileDescriptor: Int32
    public let host: String
    public let requestedPort: Int
    public let boundPort: Int

    public static func bind(
        host: String,
        port: Int,
        backlog: Int32 = 16
    ) throws -> DistributedSocketWorkerListener {
        guard (0...65_535).contains(port) else {
            throw DistributedSocketTransportError.badAddress("\(host):\(port)")
        }

        let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw DistributedSocketTransportError.socketError(operation: "socket", errno: errno)
        }
        do {
            var yes: Int32 = 1
            guard Darwin.setsockopt(
                fd,
                SOL_SOCKET,
                SO_REUSEADDR,
                &yes,
                socklen_t(MemoryLayout<Int32>.size)) == 0
            else {
                throw DistributedSocketTransportError.socketError(
                    operation: "setsockopt", errno: errno)
            }

            var address = try Self.makeIPv4Address(host: host, port: port)
            let bindResult = withUnsafePointer(to: &address) { pointer in
                pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
                    Darwin.bind(
                        fd,
                        sockaddrPointer,
                        socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            guard bindResult == 0 else {
                throw DistributedSocketTransportError.socketError(operation: "bind", errno: errno)
            }
            guard Darwin.listen(fd, backlog) == 0 else {
                throw DistributedSocketTransportError.socketError(operation: "listen", errno: errno)
            }

            let actualPort = try Self.boundPort(for: fd)
            return DistributedSocketWorkerListener(
                fileDescriptor: fd,
                host: host,
                requestedPort: port,
                boundPort: actualPort)
        } catch {
            Darwin.close(fd)
            throw error
        }
    }

    private init(
        fileDescriptor: Int32,
        host: String,
        requestedPort: Int,
        boundPort: Int
    ) {
        self.fileDescriptor = fileDescriptor
        self.host = host
        self.requestedPort = requestedPort
        self.boundPort = boundPort
    }

    deinit {
        close()
    }

    public func accept() throws -> DistributedSocketWorkerConnection {
        while true {
            let fd = Darwin.accept(fileDescriptor, nil, nil)
            if fd >= 0 {
                return DistributedSocketWorkerConnection(fileDescriptor: fd)
            }
            if errno == EINTR {
                continue
            }
            throw DistributedSocketTransportError.socketError(operation: "accept", errno: errno)
        }
    }

    public func close() {
        Darwin.shutdown(fileDescriptor, SHUT_RDWR)
        Darwin.close(fileDescriptor)
    }

    private static func makeIPv4Address(host: String, port: Int) throws -> sockaddr_in {
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(port).bigEndian

        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed == "0.0.0.0" || trimmed == "*" {
            address.sin_addr = in_addr(s_addr: INADDR_ANY)
            return address
        }
        address.sin_addr = try resolveIPv4Address(trimmed)
        return address
    }

    private static func boundPort(for fd: Int32) throws -> Int {
        var address = sockaddr_in()
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let result = withUnsafeMutablePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
                Darwin.getsockname(fd, sockaddrPointer, &length)
            }
        }
        guard result == 0 else {
            throw DistributedSocketTransportError.socketError(
                operation: "getsockname", errno: errno)
        }
        return Int(in_port_t(bigEndian: address.sin_port))
    }
}

public final class DistributedSocketWorkerConnection: @unchecked Sendable {
    private let fileDescriptor: Int32
    private var decoder = DistributedWorkerWireFrameStreamDecoder()
    private var closed = false
    private let lock = NSLock()

    public init(fileDescriptor: Int32) {
        self.fileDescriptor = fileDescriptor
    }

    deinit {
        close()
    }

    public static func connect(
        host: String,
        port: Int
    ) throws -> DistributedSocketWorkerConnection {
        guard (1...65_535).contains(port) else {
            throw DistributedSocketTransportError.badAddress("\(host):\(port)")
        }
        let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw DistributedSocketTransportError.socketError(operation: "socket", errno: errno)
        }
        do {
            var address = sockaddr_in()
            address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
            address.sin_family = sa_family_t(AF_INET)
            address.sin_port = in_port_t(port).bigEndian
            address.sin_addr = try resolveIPv4Address(host)

            let result = withUnsafePointer(to: &address) { pointer in
                pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPointer in
                    Darwin.connect(
                        fd,
                        sockaddrPointer,
                        socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            guard result == 0 else {
                throw DistributedSocketTransportError.socketError(
                    operation: "connect", errno: errno)
            }
            return DistributedSocketWorkerConnection(fileDescriptor: fd)
        } catch {
            Darwin.close(fd)
            throw error
        }
    }

    public func send(_ frame: DistributedWorkerWireFrame) throws {
        let data = try DistributedWorkerMessageCodec.encodeWireFrame(frame)
        lock.lock()
        defer { lock.unlock() }
        try writeAll(data)
    }

    public func receive() throws -> DistributedWorkerWireFrame {
        lock.lock()
        defer { lock.unlock() }
        return try receiveLocked()
    }

    public func roundTrip(
        _ request: DistributedWorkerWireFrame
    ) throws -> DistributedWorkerWireFrame? {
        lock.lock()
        defer { lock.unlock() }
        try writeAll(try DistributedWorkerMessageCodec.encodeWireFrame(request))
        switch request.message {
        case .forward:
            return try receiveLocked()
        case .hello, .helloAck, .allocate, .forwardResult, .reset, .free, .error:
            return nil
        }
    }

    public func close() {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return }
        closed = true
        Darwin.shutdown(fileDescriptor, SHUT_RDWR)
        Darwin.close(fileDescriptor)
    }

    private func receiveLocked() throws -> DistributedWorkerWireFrame {
        while true {
            if let frame = try decoder.nextFrame() {
                return frame
            }

            var buffer = [UInt8](repeating: 0, count: 64 * 1024)
            let count = buffer.withUnsafeMutableBytes { rawBuffer in
                Darwin.read(fileDescriptor, rawBuffer.baseAddress, rawBuffer.count)
            }
            if count > 0 {
                decoder.append(Data(buffer.prefix(count)))
                continue
            }
            if count == 0 {
                throw DistributedSocketTransportError.connectionClosed
            }
            if errno == EINTR {
                continue
            }
            throw DistributedSocketTransportError.socketError(operation: "read", errno: errno)
        }
    }

    private func writeAll(_ data: Data) throws {
        guard !closed else {
            throw DistributedSocketTransportError.connectionClosed
        }
        try data.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else { return }
            var offset = 0
            while offset < rawBuffer.count {
                let written = Darwin.write(
                    fileDescriptor,
                    baseAddress.advanced(by: offset),
                    rawBuffer.count - offset)
                if written > 0 {
                    offset += written
                    continue
                }
                if written == 0 {
                    throw DistributedSocketTransportError.connectionClosed
                }
                if errno == EINTR {
                    continue
                }
                throw DistributedSocketTransportError.socketError(
                    operation: "write", errno: errno)
            }
        }
    }
}

private func resolveIPv4Address(_ host: String) throws -> in_addr {
    let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else {
        throw DistributedSocketTransportError.badAddress(host)
    }
    if trimmed == "localhost" {
        return try resolveIPv4Address("127.0.0.1")
    }

    var parsed = in_addr()
    if inet_pton(AF_INET, trimmed, &parsed) == 1 {
        return parsed
    }

    guard let hostEntry = gethostbyname(trimmed),
        hostEntry.pointee.h_addrtype == AF_INET,
        let addressList = hostEntry.pointee.h_addr_list,
        let firstAddress = addressList[0]
    else {
        throw DistributedSocketTransportError.badAddress(host)
    }

    var resolved = in_addr()
    memcpy(&resolved, firstAddress, Int(hostEntry.pointee.h_length))
    return resolved
}
