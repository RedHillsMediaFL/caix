import Foundation

let caixDefaultExportsDisplayPath = "~/.caix/models/exports"

func caixDefaultExportsPath() -> String {
    NSHomeDirectory() + "/.caix/models/exports"
}

func caixExpandPath(_ path: String) -> String {
    if path == "~" { return NSHomeDirectory() }
    if path.hasPrefix("~/") { return NSHomeDirectory() + String(path.dropFirst()) }
    return path
}
