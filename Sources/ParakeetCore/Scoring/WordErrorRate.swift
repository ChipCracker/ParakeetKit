//
//  WordErrorRate.swift
//  Word error rate for transcript evaluation (benchmarks, model/quant A-B
//  comparisons). Pure Swift, no dependencies.
//
import Foundation

public enum WordErrorRate {
    /// Lowercases, strips punctuation/symbols, splits on whitespace.
    public static func normalize(_ text: String) -> [String] {
        let lowered = text.lowercased()
        var words: [String] = []
        var current = ""
        for scalar in lowered.unicodeScalars {
            if CharacterSet.whitespacesAndNewlines.contains(scalar) {
                if !current.isEmpty { words.append(current); current = "" }
            } else if CharacterSet.punctuationCharacters.contains(scalar)
                        || CharacterSet.symbols.contains(scalar) {
                continue
            } else {
                current.unicodeScalars.append(scalar)
            }
        }
        if !current.isEmpty { words.append(current) }
        return words
    }

    /// Word-level Levenshtein distance (two-row DP, O(m·n)).
    public static func distance(_ reference: [String], _ hypothesis: [String]) -> Int {
        if reference.isEmpty { return hypothesis.count }
        if hypothesis.isEmpty { return reference.count }
        var prev = Array(0...hypothesis.count)
        var curr = [Int](repeating: 0, count: hypothesis.count + 1)
        for i in 1...reference.count {
            curr[0] = i
            for j in 1...hypothesis.count {
                let substitution = prev[j - 1] + (reference[i - 1] == hypothesis[j - 1] ? 0 : 1)
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, substitution)
            }
            swap(&prev, &curr)
        }
        return prev[hypothesis.count]
    }

    /// WER = edit distance / reference length, both sides normalized.
    /// Returns 0 for an empty reference with an empty hypothesis.
    public static func wer(reference: String, hypothesis: String) -> Double {
        let ref = normalize(reference)
        let hyp = normalize(hypothesis)
        if ref.isEmpty { return hyp.isEmpty ? 0 : 1 }
        return Double(distance(ref, hyp)) / Double(ref.count)
    }
}
