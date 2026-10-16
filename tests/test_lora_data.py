from dataclasses import replace
import random
import unittest

from ai_journey.lora_data import (
    Document,
    DocumentSplit,
    LoRAExperimentError,
    synthetic_fixture,
    training_batches,
    schedule_evidence,
    heldout_targets,
    heldout_evidence,
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


class LoRADocumentWindowTests(unittest.TestCase):
    def test_training_schedule_is_local_reproducible_and_document_isolated(self):
        split = synthetic_fixture()
        controls = dict(block_size=7, batch_size=3, steps=5, seed=12)
        before = random.getstate()
        first = training_batches(split.train, split.codec(), **controls)
        self.assertEqual(before, random.getstate())
        self.assertEqual(
            first, training_batches(split.train, split.codec(), **controls)
        )
        self.assertNotEqual(
            first,
            training_batches(split.train, split.codec(), **{**controls, "seed": 13}),
        )
        docs = {doc.document_id: split.codec().encode(doc.text) for doc in split.train}
        for batch in first:
            for row in batch:
                tokens = docs[row.document_id]
                self.assertEqual(row.inputs, tokens[row.start : row.start + 7])
                self.assertEqual(row.targets, tokens[row.start + 1 : row.start + 8])
        self.assertEqual(schedule_evidence(first)["target_tokens"], 105)

    def test_holdout_covers_short_remainder_and_long_documents_exactly_once(self):
        split = synthetic_fixture()
        documents = (Document("short", "ca"), Document("long", "calm cedar.\n"))
        targets = heldout_targets(documents, split.codec(), block_size=4)
        expected = [
            (doc.document_id, i) for doc in documents for i in range(1, len(doc.text))
        ]
        self.assertEqual(
            [(row.document_id, row.target_index) for row in targets], expected
        )
        for row in targets:
            text = next(
                doc.text for doc in documents if doc.document_id == row.document_id
            )
            self.assertEqual(
                split.codec().decode(row.context),
                text[max(0, row.target_index - 4) : row.target_index],
            )
            self.assertEqual(split.codec().tokens[row.target], text[row.target_index])
        evidence = heldout_evidence(documents, split.codec(), block_size=4)
        self.assertEqual(evidence["target_tokens"], len(expected))
        self.assertEqual(
            sum(row["target_tokens"] for row in evidence["per_document"]), len(expected)
        )

    def test_window_controls_fail_before_sampling(self):
        split = synthetic_fixture()
        controls = dict(block_size=8, batch_size=2, steps=2, seed=1)
        for field, value in (
            ("block_size", True),
            ("batch_size", 9),
            ("steps", 65),
            ("seed", -1),
        ):
            with self.subTest(field=field), self.assertRaises(LoRAExperimentError):
                training_batches(
                    split.train, split.codec(), **{**controls, field: value}
                )
        with self.assertRaisesRegex(LoRAExperimentError, "full block"):
            training_batches((Document("short", "ca"),), split.codec(), **controls)
        for docs in ((), (split.train[0], split.train[0])):
            with self.assertRaises(LoRAExperimentError):
                heldout_targets(docs, split.codec(), block_size=4)
