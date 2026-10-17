import unittest
from dataclasses import replace

from ai_journey.sft_data import ChatMessage, Conversation, SFTDataError, ToolCall


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


if __name__ == "__main__":
    unittest.main()
