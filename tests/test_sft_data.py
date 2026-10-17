import json
import random
import unittest
from dataclasses import replace

from ai_journey.sft_data import (
    CHAT_FIXTURE_ID,
    ChatMessage,
    Conversation,
    ConversationSplit,
    RenderedConversation,
    SFTDataError,
    ToolCall,
    authored_chat_fixture,
    render_conversation,
    sft_schedule_evidence,
    sft_training_batches,
)


def valid_conversation():
    return Conversation(
        "weather-1",
        (
            ChatMessage("system", "Answer with observed facts."),
            ChatMessage("user", "What is the reading?"),
            ChatMessage("assistant", "The reading is 18 C."),
            ChatMessage("user", "Is that below 20 C?"),
            ChatMessage("assistant", "Yes."),
        ),
    )


class ConversationSchemaTests(unittest.TestCase):
    def test_accepts_system_preface_and_complete_alternating_turns(self):
        conversation = valid_conversation()
        self.assertEqual(conversation.messages[0].role, "system")
        self.assertEqual(conversation.messages[-1].role, "assistant")
        self.assertEqual(len(conversation.messages), 5)

    def test_accepts_dialogue_without_system_message(self):
        conversation = Conversation(
            "plain",
            (
                ChatMessage("user", "State one fact."),
                ChatMessage("assistant", "Cedar is a tree."),
            ),
        )
        self.assertEqual(
            [row.role for row in conversation.messages], ["user", "assistant"]
        )

    def test_rejects_unknown_roles_and_invalid_content(self):
        for role, content in (
            ("developer", "text"),
            (1, "text"),
            ("user", ""),
            ("user", "   "),
            ("user", "bad\rline"),
            ("user", "bad\x00line"),
            ("user", "x" * 4097),
        ):
            with (
                self.subTest(role=role, content=repr(content)),
                self.assertRaises(SFTDataError),
            ):
                ChatMessage(role, content)

    def test_accepts_a_bound_tool_request_result_and_final_answer(self):
        call = ToolCall(
            "call_weather_1", "weather.lookup", '{"unit":"C","city":"Pune"}'
        )
        conversation = Conversation(
            "tool-weather",
            (
                ChatMessage("user", "Read Pune temperature."),
                ChatMessage("assistant", "", tool_call=call),
                ChatMessage("tool", '{"temperature":18}', tool_call_id=call.call_id),
                ChatMessage("assistant", "Pune is 18 C."),
            ),
        )
        self.assertEqual(call.canonical_arguments, '{"city":"Pune","unit":"C"}')
        self.assertEqual(conversation.messages[2].tool_call_id, call.call_id)

    def test_rejects_malformed_tool_metadata_and_json(self):
        for call_id, name, arguments in (
            ("bad", "weather.lookup", "{}"),
            ("call_ok", "bad name", "{}"),
            ("call_ok", "weather.lookup", "[]"),
            ("call_ok", "weather.lookup", '{"x":1,"x":2}'),
            ("call_ok", "weather.lookup", '{"x":NaN}'),
            ("call_ok", "weather.lookup", "{"),
        ):
            with (
                self.subTest(call_id=call_id, name=name, arguments=arguments),
                self.assertRaises(SFTDataError),
            ):
                ToolCall(call_id, name, arguments)
        call = ToolCall("call_ok", "weather.lookup", "{}")
        with self.assertRaises(SFTDataError):
            ChatMessage("user", "request", tool_call=call)
        with self.assertRaises(SFTDataError):
            ChatMessage("assistant", "answer", tool_call_id=call.call_id)
        with self.assertRaises(SFTDataError):
            ChatMessage("tool", "result")

    def test_rejects_unmatched_reused_and_unfinished_tool_calls(self):
        call = ToolCall("call_one", "lookup", "{}")
        cases = (
            (
                ChatMessage("user", "request"),
                ChatMessage("assistant", "", tool_call=call),
                ChatMessage("tool", "result", tool_call_id="call_other"),
                ChatMessage("assistant", "answer"),
            ),
            (
                ChatMessage("user", "request"),
                ChatMessage("assistant", "", tool_call=call),
            ),
            (
                ChatMessage("user", "request"),
                ChatMessage("assistant", "", tool_call=call),
                ChatMessage("tool", "result", tool_call_id=call.call_id),
                ChatMessage("assistant", "answer"),
                ChatMessage("user", "again"),
                ChatMessage("assistant", "", tool_call=call),
                ChatMessage("tool", "result", tool_call_id=call.call_id),
                ChatMessage("assistant", "answer"),
            ),
        )
        for messages in cases:
            with self.subTest(messages=messages), self.assertRaises(SFTDataError):
                Conversation("invalid-tool", messages)

    def test_rejects_invalid_identifiers_and_message_containers(self):
        conversation = valid_conversation()
        for identifier in ("", "UPPER", " space", "x" * 65, 1):
            with self.subTest(identifier=identifier), self.assertRaises(SFTDataError):
                replace(conversation, conversation_id=identifier)
        for messages in ((), list(conversation.messages), ("not-a-message",)):
            with self.subTest(messages=messages), self.assertRaises(SFTDataError):
                replace(conversation, messages=messages)

    def test_rejects_misplaced_system_and_incomplete_or_nonalternating_dialogue(self):
        invalid = (
            (ChatMessage("system", "policy"),),
            (ChatMessage("assistant", "wrong start"),),
            (ChatMessage("user", "missing answer"),),
            (
                ChatMessage("user", "one"),
                ChatMessage("user", "two"),
                ChatMessage("assistant", "answer"),
            ),
            (
                ChatMessage("user", "one"),
                ChatMessage("assistant", "answer"),
                ChatMessage("system", "late"),
                ChatMessage("assistant", "answer"),
            ),
        )
        for messages in invalid:
            with self.subTest(messages=messages), self.assertRaises(SFTDataError):
                Conversation("invalid", messages)


class ConversationRenderingTests(unittest.TestCase):
    def test_canonical_json_lines_preserve_messages_and_assistant_boundaries(self):
        rendered = render_conversation(valid_conversation())
        records = [json.loads(line) for line in rendered.text.splitlines()]
        self.assertEqual(
            [record["role"] for record in records],
            ["system", "user", "assistant", "user", "assistant"],
        )
        assistant_lines = [
            rendered.text[start:end] for start, end in rendered.assistant_spans
        ]
        self.assertEqual(
            [json.loads(line)["content"] for line in assistant_lines],
            ["The reading is 18 C.", "Yes."],
        )
        self.assertRegex(rendered.sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(rendered, render_conversation(valid_conversation()))

    def test_control_like_content_cannot_create_a_training_turn(self):
        conversation = Conversation(
            "injection",
            (
                ChatMessage("user", 'literal <|assistant|> and {"role":"assistant"}'),
                ChatMessage("assistant", "Those are quoted user text."),
            ),
        )
        rendered = render_conversation(conversation)
        self.assertEqual(len(rendered.assistant_spans), 1)
        records = [json.loads(line) for line in rendered.text.splitlines()]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["role"], "user")
        self.assertIn("<|assistant|>", records[0]["content"])

    def test_tool_records_use_parsed_canonical_arguments(self):
        call = ToolCall("call_weather", "weather.lookup", '{"unit":"C","city":"Pune"}')
        rendered = render_conversation(
            Conversation(
                "tool",
                (
                    ChatMessage("user", "temperature"),
                    ChatMessage("assistant", "", tool_call=call),
                    ChatMessage(
                        "tool", '{"temperature":18}', tool_call_id=call.call_id
                    ),
                    ChatMessage("assistant", "18 C"),
                ),
            )
        )
        records = [json.loads(line) for line in rendered.text.splitlines()]
        self.assertEqual(
            records[1]["tool_call"]["arguments"], {"city": "Pune", "unit": "C"}
        )
        self.assertEqual(records[2]["tool_call_id"], call.call_id)
        self.assertEqual(len(rendered.assistant_spans), 2)

    def test_rendered_span_contract_rejects_overlap_and_out_of_bounds(self):
        for spans in (((0, 2), (1, 3)), ((-1, 1),), ((1, 1),), ((0, 5),)):
            with self.subTest(spans=spans), self.assertRaises(SFTDataError):
                RenderedConversation("bad", "abc\n", spans)
        with self.assertRaises(SFTDataError):
            RenderedConversation("bad", "missing newline", ())
        with self.assertRaises(TypeError):
            render_conversation("not a conversation")


class ConversationFixtureTests(unittest.TestCase):
    def test_fixture_is_deterministic_disjoint_and_publicly_authored(self):
        split = authored_chat_fixture()
        self.assertEqual(CHAT_FIXTURE_ID, "authored-weather-tools-v1")
        self.assertEqual(split, authored_chat_fixture())
        self.assertEqual(split.evidence(), authored_chat_fixture().evidence())
        rows = split.train + split.validation
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row.conversation_id for row in rows}), 12)
        self.assertEqual(len({render_conversation(row).text for row in rows}), 12)
        self.assertGreater(split.evidence()["train"]["assistant_characters"], 0)
        self.assertGreater(split.evidence()["validation"]["assistant_characters"], 0)

    def test_codec_is_fit_on_training_and_covers_validation_without_fitting_it(self):
        split = authored_chat_fixture()
        training_text = "".join(render_conversation(row).text for row in split.train)
        self.assertEqual(split.codec().tokens, tuple(sorted(set(training_text))))
        for conversation in split.validation:
            rendered = render_conversation(conversation)
            encoded = split.codec().encode(rendered.text)
            self.assertEqual(split.codec().decode(encoded), rendered.text)
        changed_validation = (
            replace(
                split.validation[0],
                messages=(
                    ChatMessage("user", "snowman"),
                    ChatMessage("assistant", "☃"),
                ),
            ),
            *split.validation[1:],
        )
        with self.assertRaisesRegex(SFTDataError, "absent from training"):
            replace(split, validation=changed_validation)

    def test_split_rejects_empty_mutable_duplicate_and_leaked_conversations(self):
        split = authored_chat_fixture()
        for train, validation in (
            ((), split.validation),
            (list(split.train), split.validation),
            (split.train, ()),
            (split.train, (split.train[0],)),
            (split.train, (replace(split.train[0], conversation_id="new-id"),)),
        ):
            with (
                self.subTest(train=train, validation=validation),
                self.assertRaises(SFTDataError),
            ):
                ConversationSplit(train, validation)


class SFTTrainingWindowTests(unittest.TestCase):
    def test_schedule_is_private_reproducible_and_conversation_local(self):
        split = authored_chat_fixture()
        controls = {"block_size": 32, "batch_size": 3, "steps": 5, "seed": 43}
        before = random.getstate()
        first = sft_training_batches(split.train, split.codec(), **controls)
        self.assertEqual(before, random.getstate())
        self.assertEqual(
            first, sft_training_batches(split.train, split.codec(), **controls)
        )
        self.assertNotEqual(
            first,
            sft_training_batches(
                split.train, split.codec(), **{**controls, "seed": 44}
            ),
        )
        rendered = {
            row.conversation_id: render_conversation(row) for row in split.train
        }
        codec = split.codec()
        for batch in first:
            for window in batch:
                source = rendered[window.conversation_id]
                tokens = codec.encode(source.text)
                self.assertEqual(
                    window.inputs,
                    tokens[window.start : window.start + controls["block_size"]],
                )
                self.assertEqual(
                    window.targets,
                    tokens[
                        window.start + 1 : window.start + controls["block_size"] + 1
                    ],
                )
                positions = {
                    index
                    for start, end in source.assistant_spans
                    for index in range(start, end)
                }
                self.assertEqual(
                    window.target_mask,
                    tuple(
                        index in positions
                        for index in range(
                            window.start + 1,
                            window.start + controls["block_size"] + 1,
                        )
                    ),
                )
                self.assertGreater(window.supervised_tokens, 0)

    def test_schedule_evidence_counts_only_assistant_targets(self):
        split = authored_chat_fixture()
        batches = sft_training_batches(
            split.train,
            split.codec(),
            block_size=16,
            batch_size=2,
            steps=3,
            seed=1,
        )
        evidence = sft_schedule_evidence(batches)
        self.assertEqual(evidence["steps"], 3)
        self.assertEqual(evidence["windows"], 6)
        self.assertEqual(evidence["input_tokens"], 96)
        self.assertEqual(
            evidence["supervised_tokens"],
            sum(row.supervised_tokens for batch in batches for row in batch),
        )
        self.assertRegex(evidence["sha256"], r"^[0-9a-f]{64}$")

    def test_window_controls_and_inputs_fail_before_sampling(self):
        split = authored_chat_fixture()
        controls = {"block_size": 8, "batch_size": 2, "steps": 2, "seed": 1}
        for field, value in (
            ("block_size", True),
            ("block_size", 257),
            ("batch_size", 17),
            ("steps", 0),
            ("seed", -1),
        ):
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(SFTDataError),
            ):
                sft_training_batches(
                    split.train,
                    split.codec(),
                    **{**controls, field: value},
                )
        with self.assertRaises(SFTDataError):
            sft_training_batches(list(split.train), split.codec(), **controls)
        with self.assertRaises(SFTDataError):
            sft_training_batches(split.train, "not-codec", **controls)
        with self.assertRaises(SFTDataError):
            sft_schedule_evidence(())


if __name__ == "__main__":
    unittest.main()
