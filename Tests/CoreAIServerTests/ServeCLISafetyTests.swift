import Foundation
import XCTest

final class ServeCLISafetyTests: XCTestCase {
    func testOmittedEagleUnrolledFlagHasNoImplicitAssetDefault() throws {
        let mainSource = try pipelineCLIMainSource()
        let declaration = try XCTUnwrap(
            mainSource.split(separator: "\n", omittingEmptySubsequences: false)
                .map(String.init)
                .first(where: { $0.contains("var eagleUnrolled: String?") }))

        XCTAssertEqual(
            declaration.trimmingCharacters(in: .whitespaces),
            "var eagleUnrolled: String? = nil",
            "caix serve must not select any unrolled draft unless --eagle-unrolled is explicit")
    }

    func testExplicitEagleUnrolledFlagRetainsExistenceGate() throws {
        let mainSource = try pipelineCLIMainSource()

        XCTAssertTrue(
            mainSource.contains("case \"--eagle-unrolled\": eagleUnrolled = value(arg)"),
            "serve must only populate the unrolled path from the explicit CLI flag")
        XCTAssertTrue(
            mainSource.contains(
                "let unrolled = eagleUnrolled.flatMap { fm.fileExists(atPath: $0) ? $0 : nil }"),
            "serve must not enable a nonexistent explicitly supplied unrolled bundle")
    }

    func testServeHelpDocumentsExplicitUnrolledSelectionHasNoDefault() throws {
        XCTAssertTrue(
            try pipelineCLIMainSource().contains(
                "--eagle-unrolled <dir>  Explicit unrolled EAGLE draft .aimodel bundle (no default)"))
    }

    private func pipelineCLIMainSource() throws -> String {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(
            contentsOf: repositoryRoot.appendingPathComponent("Sources/PipelineCLI/main.swift"),
            encoding: .utf8)
    }
}
