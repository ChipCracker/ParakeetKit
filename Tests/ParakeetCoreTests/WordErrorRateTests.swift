import XCTest
import ParakeetCore

final class WordErrorRateTests: XCTestCase {

    func testNormalizeStripsPunctuationAndCase() {
        XCTAssertEqual(WordErrorRate.normalize("And so, my fellow Americans:"),
                       ["and", "so", "my", "fellow", "americans"])
        XCTAssertEqual(WordErrorRate.normalize("  weiß—gefärbt!  "), ["weißgefärbt"])
        XCTAssertEqual(WordErrorRate.normalize(""), [])
    }

    func testDistance() {
        XCTAssertEqual(WordErrorRate.distance([], []), 0)
        XCTAssertEqual(WordErrorRate.distance(["a", "b"], []), 2)
        XCTAssertEqual(WordErrorRate.distance([], ["a"]), 1)
        XCTAssertEqual(WordErrorRate.distance(["a", "b", "c"], ["a", "b", "c"]), 0)
        XCTAssertEqual(WordErrorRate.distance(["a", "b", "c"], ["a", "x", "c"]), 1)   // sub
        XCTAssertEqual(WordErrorRate.distance(["a", "b", "c"], ["a", "c"]), 1)        // del
        XCTAssertEqual(WordErrorRate.distance(["a", "c"], ["a", "b", "c"]), 1)        // ins
    }

    func testWER() {
        XCTAssertEqual(WordErrorRate.wer(reference: "ask not what", hypothesis: "ask not what"), 0)
        XCTAssertEqual(WordErrorRate.wer(reference: "Ask not, what!", hypothesis: "ask not what"), 0)
        XCTAssertEqual(WordErrorRate.wer(reference: "a b c d", hypothesis: "a b c x"), 0.25)
        XCTAssertEqual(WordErrorRate.wer(reference: "", hypothesis: ""), 0)
        XCTAssertEqual(WordErrorRate.wer(reference: "", hypothesis: "noise"), 1)
        XCTAssertEqual(WordErrorRate.wer(reference: "a b", hypothesis: ""), 1)
    }
}
