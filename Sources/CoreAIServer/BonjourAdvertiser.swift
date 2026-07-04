import Foundation

final class BonjourAdvertiser: NSObject, NetServiceDelegate {
    private let service: NetService?
    private let verbose: Bool

    init(host: String, port: Int, caixVersion: String, verbose: Bool) {
        self.verbose = verbose
        guard Self.shouldAdvertise(host: host, port: port) else {
            self.service = nil
            super.init()
            return
        }

        let machine = Host.current().localizedName ?? Host.current().name ?? "Mac"
        let serviceName = "caix \(machine)"
        let service = NetService(domain: "local.", type: "_caix._tcp.", name: serviceName, port: Int32(port))
        service.setTXTRecord(NetService.data(fromTXTRecord: [
            "api": Data("openai".utf8),
            "path": Data("/".utf8),
            "version": Data(caixVersion.utf8),
            "host": Data(host.utf8),
        ]))
        self.service = service
        super.init()
        service.delegate = self
    }

    func start() {
        service?.publish()
    }

    func stop() {
        service?.stop()
    }

    func netServiceDidPublish(_ sender: NetService) {
        guard verbose else { return }
        FileHandle.standardError.write(
            Data("[bonjour] published \(sender.name) \(sender.type) port \(sender.port)\n".utf8))
    }

    func netService(_ sender: NetService, didNotPublish errorDict: [String: NSNumber]) {
        FileHandle.standardError.write(
            Data("[bonjour] publish failed for \(sender.type): \(errorDict)\n".utf8))
    }

    private static func shouldAdvertise(host: String, port: Int) -> Bool {
        guard port > 0 else { return false }
        let env = ProcessInfo.processInfo.environment
        if let raw = env["CAIX_BONJOUR"]?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
            if ["0", "false", "off", "no"].contains(raw) { return false }
            if ["1", "true", "on", "yes"].contains(raw) { return true }
        }
        let normalized = host.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized != "127.0.0.1"
            && normalized != "::1"
            && normalized != "localhost"
            && !normalized.isEmpty
    }
}
