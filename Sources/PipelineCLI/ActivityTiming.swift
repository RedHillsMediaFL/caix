import Foundation

func activityTiming(_ row: [String: Any]) -> String {
    var parts: [String] = []
    if let ttft = numberFromKeys(row, "firstTokenMs", "first_token_ms", "ttftMs", "ttft_ms") {
        parts.append("ttft \(Int(ttft.rounded()))ms")
    }
    if let prefill = numberFromKeys(row, "prefillMs", "prefill_ms") {
        parts.append("prefill \(Int(prefill.rounded()))ms")
    }
    if let decode = numberFromKeys(row, "decodeMs", "decode_ms") {
        parts.append("decode \(Int(decode.rounded()))ms")
    }
    if let tps = numberFromKeys(row, "decodeTokensPerSecond", "decode_tokens_per_second") {
        parts.append(String(format: "%.1f tok/s", tps))
    }
    return parts.isEmpty ? "" : " [\(parts.joined(separator: ", "))]"
}

private func numberFromKeys(_ row: [String: Any], _ keys: String...) -> Double? {
    for key in keys {
        if let value = doubleFromNumber(row[key]) { return value }
    }
    return nil
}

private func doubleFromNumber(_ value: Any?) -> Double? {
    if let double = value as? Double { return double }
    if let int = value as? Int { return Double(int) }
    if let number = value as? NSNumber { return number.doubleValue }
    if let string = value as? String { return Double(string) }
    return nil
}
