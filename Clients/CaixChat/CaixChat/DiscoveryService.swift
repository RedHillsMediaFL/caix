import Darwin
import Foundation

@MainActor
final class DiscoveryService {
    private let bonjourBrowser = BonjourBrowser()

    func discover() async -> [ServerEndpoint] {
        async let bonjour = bonjourBrowser.browse(timeout: 2.4)
        async let network = ActiveNetworkDiscovery.scan()

        let bonjourCandidates = await bonjour
        let networkServers = await network
        let bonjourServers = await ActiveNetworkDiscovery.probe(
            urls: bonjourCandidates.map(\.baseURL),
            source: .bonjour
        )
        return Self.merged(networkServers + bonjourServers)
    }

    private static func merged(_ endpoints: [ServerEndpoint]) -> [ServerEndpoint] {
        var byID: [String: ServerEndpoint] = [:]
        for endpoint in endpoints {
            if let existing = byID[endpoint.id] {
                var merged = existing
                merged.source = existing.source == .manual ? .manual : endpoint.source
                merged.modelCount = max(existing.modelCount, endpoint.modelCount)
                merged.isReachable = existing.isReachable || endpoint.isReachable
                merged.lastSeen = max(existing.lastSeen, endpoint.lastSeen)
                if merged.detail == nil { merged.detail = endpoint.detail }
                byID[endpoint.id] = merged
            } else {
                byID[endpoint.id] = endpoint
            }
        }
        return byID.values.sorted {
            if $0.isReachable != $1.isReachable { return $0.isReachable && !$1.isReachable }
            return $0.name.localizedStandardCompare($1.name) == .orderedAscending
        }
    }
}

private final class BonjourBrowser: NSObject, NetServiceBrowserDelegate, NetServiceDelegate {
    private var browsers: [NetServiceBrowser] = []
    private var services: [NetService] = []
    private var endpoints: [ServerEndpoint] = []
    private var continuation: CheckedContinuation<[ServerEndpoint], Never>?

    @MainActor
    func browse(timeout: TimeInterval) async -> [ServerEndpoint] {
        finish()
        return await withCheckedContinuation { continuation in
            self.continuation = continuation
            self.endpoints = []
            self.services = []
            let serviceTypes = ["_caix._tcp.", "_http._tcp.", "_https._tcp."]
            self.browsers = serviceTypes.map { type in
                let browser = NetServiceBrowser()
                browser.delegate = self
                browser.searchForServices(ofType: type, inDomain: "local.")
                return browser
            }
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
                self.finish()
            }
        }
    }

    @MainActor
    private func finish() {
        guard let continuation else { return }
        browsers.forEach { $0.stop() }
        browsers = []
        services = []
        self.continuation = nil
        continuation.resume(returning: endpoints)
    }

    func netServiceBrowser(_ browser: NetServiceBrowser, didFind service: NetService, moreComing: Bool) {
        services.append(service)
        service.delegate = self
        service.resolve(withTimeout: 2)
    }

    func netServiceDidResolveAddress(_ sender: NetService) {
        guard sender.port > 0 else { return }
        let scheme = sender.type.contains("_https") ? "https" : "http"
        let host = (sender.hostName?.trimmingCharacters(in: CharacterSet(charactersIn: "."))).flatMap {
            $0.isEmpty ? nil : $0
        } ?? sender.name
        guard let url = ServerEndpoint.normalizedURL(from: "\(scheme)://\(host):\(sender.port)") else {
            return
        }
        let endpoint = ServerEndpoint(
            baseURL: url,
            source: .bonjour,
            name: sender.name,
            detail: sender.type
        )
        if !endpoints.contains(where: { $0.id == endpoint.id }) {
            endpoints.append(endpoint)
        }
    }
}

enum ActiveNetworkDiscovery {
    static func scan() async -> [ServerEndpoint] {
        await probe(urls: NetworkCandidateBuilder.candidateURLs(), source: .network)
    }

    static func probe(urls: [URL], source: ServerSource, maxConcurrent: Int = 32) async -> [ServerEndpoint] {
        guard !urls.isEmpty else { return [] }
        let limit = max(1, maxConcurrent)
        return await withTaskGroup(of: ServerEndpoint?.self) { group in
            var iterator = urls.makeIterator()

            for _ in 0..<limit {
                guard let url = iterator.next() else { break }
                group.addTask { await probe(url: url, source: source) }
            }

            var endpoints: [ServerEndpoint] = []
            while let endpoint = await group.next() {
                if let endpoint {
                    endpoints.append(endpoint)
                }
                if let url = iterator.next() {
                    group.addTask { await probe(url: url, source: source) }
                }
            }
            return endpoints
        }
    }

    private static func probe(url: URL, source: ServerSource) async -> ServerEndpoint? {
        do {
            let models = try await CaixClient(baseURL: url, timeout: 1.2).fetchModels()
            return ServerEndpoint(
                baseURL: url,
                source: source,
                modelCount: models.count,
                isReachable: true,
                lastSeen: Date()
            )
        } catch {
            return nil
        }
    }
}

enum NetworkCandidateBuilder {
    static func candidateURLs() -> [URL] {
        var raw = Set<String>()
        raw.insert("http://127.0.0.1:1237")
        raw.insert("http://localhost:1237")
        raw.insert("http://caix.local:1237")
        raw.insert("http://caix:1237")

        for interface in ipv4Interfaces() {
            for host in hostCandidates(around: interface) {
                raw.insert("http://\(host):1237")
            }
        }

        return raw.compactMap(ServerEndpoint.normalizedURL(from:)).sorted {
            $0.absoluteString < $1.absoluteString
        }
    }

    static func hostCandidates(around interface: IPv4Interface) -> [String] {
        let octets = octets(interface.address)
        guard octets[0] != 0, octets[0] != 127, octets[0] != 169 else { return [] }

        let mask = interface.netmask
        let tailnet = interface.name.hasPrefix("utun") || (octets[0] == 100 && (64...127).contains(octets[1]))
        let subnetSize = UInt64(~mask) + 1
        let scanMask: UInt32
        if tailnet || subnetSize > 256 || mask == UInt32.max {
            scanMask = 0xFFFF_FF00
        } else {
            scanMask = mask
        }

        let network = interface.address & scanMask
        let broadcast = network | ~scanMask
        guard broadcast > network else { return [] }

        var hosts: [String] = []
        for value in (network + 1)..<broadcast {
            if value == interface.address { continue }
            hosts.append(dotted(value))
            if hosts.count >= 254 { break }
        }
        return hosts
    }

    private static func ipv4Interfaces() -> [IPv4Interface] {
        var pointer: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&pointer) == 0, let first = pointer else { return [] }
        defer { freeifaddrs(pointer) }

        var interfaces: [IPv4Interface] = []
        var cursor: UnsafeMutablePointer<ifaddrs>? = first
        while let current = cursor {
            defer { cursor = current.pointee.ifa_next }
            let flags = Int32(current.pointee.ifa_flags)
            guard flags & IFF_UP != 0,
                  flags & IFF_RUNNING != 0,
                  flags & IFF_LOOPBACK == 0,
                  let addressPointer = current.pointee.ifa_addr,
                  addressPointer.pointee.sa_family == UInt8(AF_INET),
                  let maskPointer = current.pointee.ifa_netmask
            else {
                continue
            }

            let name = String(cString: current.pointee.ifa_name)
            let address = addressPointer.withMemoryRebound(to: sockaddr_in.self, capacity: 1) {
                UInt32(bigEndian: $0.pointee.sin_addr.s_addr)
            }
            let mask = maskPointer.withMemoryRebound(to: sockaddr_in.self, capacity: 1) {
                UInt32(bigEndian: $0.pointee.sin_addr.s_addr)
            }
            interfaces.append(IPv4Interface(name: name, address: address, netmask: mask))
        }
        return interfaces
    }

    private static func octets(_ value: UInt32) -> [UInt32] {
        [
            (value >> 24) & 0xff,
            (value >> 16) & 0xff,
            (value >> 8) & 0xff,
            value & 0xff
        ]
    }

    private static func dotted(_ value: UInt32) -> String {
        let parts = octets(value).map(String.init)
        return parts.joined(separator: ".")
    }
}

struct IPv4Interface {
    var name: String
    var address: UInt32
    var netmask: UInt32
}
