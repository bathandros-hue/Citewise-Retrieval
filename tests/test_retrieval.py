import unittest

from citewise import Document, HybridIndex, evaluate
from citewise.engine import chunk_document


DOCS = [
    Document("pump", "A motor shaft rotates the centrifugal pump impeller. Pump flow is measured."),
    Document("attention", "A transformer uses attention with query key and value vectors."),
    Document("solar", "Solar thermal collectors transfer radiation into heat."),
]


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.index = HybridIndex(DOCS, chunk_size=50, overlap=4)

    def test_find_pump(self):
        self.assertEqual(self.index.search("centrifugal pump impeller", 1)[0].passage.source, "pump")

    def test_typo_robustness(self):
        self.assertEqual(self.index.search("transfomer atention query", 1)[0].passage.source, "attention")

    def test_offsets_exact(self):
        doc = Document("d", "First paragraph. Second paragraph, with words.")
        for p in chunk_document(doc, size=3, overlap=1):
            self.assertEqual(doc.text[p.start : p.end], p.text)

    def test_eval(self):
        results = evaluate(self.index, [("impeller motor", "pump"), ("solar heat", "solar")], k=1)
        self.assertEqual(results["recall@1"], 1.0)
        self.assertEqual(results["mrr@1"], 1.0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            chunk_document(DOCS[0], size=3, overlap=3)
        with self.assertRaises(ValueError):
            HybridIndex([DOCS[0], DOCS[0]])
        with self.assertRaises(ValueError):
            self.index.search("pump", k=0)
        self.assertEqual(self.index.search("    "), [])


if __name__ == "__main__":
    unittest.main()
