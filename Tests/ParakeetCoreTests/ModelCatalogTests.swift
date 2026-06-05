import XCTest
import ParakeetCore

final class ModelCatalogTests: XCTestCase {
    func testBuiltinURLAndMetadata() {
        let spec = ParakeetModelCatalog.q4_K
        XCTAssertEqual(spec.fileName, "parakeet-tdt-0.6b-v3-q4_k.gguf")
        XCTAssertEqual(spec.downloadURL.absoluteString,
                       "https://huggingface.co/cstr/parakeet-tdt-0.6b-v3-GGUF/resolve/main/parakeet-tdt-0.6b-v3-q4_k.gguf?download=true")
        XCTAssertEqual(spec.family, "Parakeet TDT 0.6B v3")
        XCTAssertEqual(ParakeetModelCatalog.recommended.id, "parakeet-tdt-0.6b-v3-q4_k")
    }

    func testF16FilenameHasNoQuantSuffix() {
        XCTAssertEqual(ParakeetModelCatalog.f16.fileName, "parakeet-tdt-0.6b-v3.gguf")
    }

    func testDefaultCatalogHasFourBuiltins() {
        let catalog = ParakeetModelCatalog()
        XCTAssertEqual(catalog.all.count, 4)
        XCTAssertNotNil(catalog.spec(id: "parakeet-tdt-0.6b-v3-q8_0"))
    }

    func testRegisterCustomModel() {
        let catalog = ParakeetModelCatalog()
        let custom = ParakeetModelSpec.huggingFace(
            id: "custom-asr", displayName: "Q5_K", family: "Custom",
            quantization: .custom("Q5_K"), repo: "me/custom-GGUF",
            fileName: "custom.gguf", approxBytes: 100_000_000)
        catalog.register(custom)
        XCTAssertEqual(catalog.all.count, 5)
        XCTAssertEqual(catalog.spec(id: "custom-asr")?.displayName, "Q5_K")
    }
}
