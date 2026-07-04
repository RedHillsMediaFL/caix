import Foundation
import UIKit

struct ImageAttachment: Identifiable, Equatable {
    var id = UUID()
    var jpegData: Data
    var previewData: Data
    var mimeType: String

    init(data: Data, maxDimension: CGFloat = 1280) throws {
        guard let image = UIImage(data: data) else {
            throw AttachmentError.invalidImage
        }
        let scaled = image.caixScaled(maxDimension: maxDimension)
        guard let jpeg = scaled.jpegData(compressionQuality: 0.82),
              let preview = scaled.caixScaled(maxDimension: 360).jpegData(compressionQuality: 0.78)
        else {
            throw AttachmentError.encodingFailed
        }
        self.jpegData = jpeg
        self.previewData = preview
        self.mimeType = "image/jpeg"
    }

    var dataURL: String {
        "data:\(mimeType);base64,\(jpegData.base64EncodedString())"
    }

    var previewImage: UIImage? {
        UIImage(data: previewData)
    }

    enum AttachmentError: LocalizedError {
        case invalidImage
        case encodingFailed

        var errorDescription: String? {
            switch self {
            case .invalidImage: return "The selected image could not be opened."
            case .encodingFailed: return "The selected image could not be prepared."
            }
        }
    }
}

private extension UIImage {
    func caixScaled(maxDimension: CGFloat) -> UIImage {
        let largestSide = max(size.width, size.height)
        guard largestSide > maxDimension else { return self }
        let scale = maxDimension / largestSide
        let newSize = CGSize(width: size.width * scale, height: size.height * scale)
        let renderer = UIGraphicsImageRenderer(size: newSize)
        return renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: newSize))
        }
    }
}
