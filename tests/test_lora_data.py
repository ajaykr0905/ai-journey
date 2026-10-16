from dataclasses import replace
import unittest

from ai_journey.lora_data import (
    Document,
    DocumentSplit,
    LoRAExperimentError,
    synthetic_fixture,
)


class LoRAFixtureTests(unittest.TestCase):
    def test_fixture_is_reproducible_with_disjoint_ids_and_texts(self):
        split = synthetic_fixture()
        self.assertEqual(split, synthetic_fixture())
        self.assertEqual(split.evidence(), synthetic_fixture().evidence())
        docs = split.pretrain + split.train + split.validation
        self.assertEqual(len(docs), 20)
        self.assertEqual(len({doc.text for doc in docs}), 20)
        self.assertEqual(len({doc.document_id for doc in docs}), 20)
        self.assertEqual(split.evidence()["validation"]["targets"], 376)

    def test_training_codec_covers_holdout_without_fitting_it(self):
        split = synthetic_fixture()
        expected = tuple(
            sorted(set("".join(doc.text for doc in split.pretrain + split.train)))
        )
        self.assertEqual(split.codec().tokens, expected)
        for doc in split.validation:
            self.assertEqual(
                split.codec().decode(split.codec().encode(doc.text)), doc.text
            )
        with self.assertRaisesRegex(LoRAExperimentError, "unseen character"):
            replace(split, validation=(Document("new", "☃ unseen"),))

    def test_rejects_alias_ids_exact_text_leakage_and_empty_partitions(self):
        split = synthetic_fixture()
        for validation in (
            (),
            list(split.validation),
            (Document(split.train[0].document_id, "new text"),),
            (Document("new", split.train[0].text),),
        ):
            with self.subTest(validation=validation):
                with self.assertRaises(LoRAExperimentError):
                    replace(split, validation=validation)
        with self.assertRaises(LoRAExperimentError):
            DocumentSplit(split.pretrain, (), split.validation)

    def test_evidence_identifies_content_and_order(self):
        split = synthetic_fixture()
        reordered = replace(split, train=split.train[::-1])
        changed = replace(
            split,
            train=(
                replace(split.train[0], text=split.train[0].text + "\n"),
                *split.train[1:],
            ),
        )
        self.assertNotEqual(split.evidence()["train"], reordered.evidence()["train"])
        self.assertNotEqual(split.evidence()["train"], changed.evidence()["train"])

    def test_document_validation(self):
        for doc_id, text in (("", "abc"), (" ", "abc"), (1, "abc"), ("id", "a")):
            with self.subTest(doc_id=doc_id, text=text):
                with self.assertRaises(LoRAExperimentError):
                    Document(doc_id, text)
