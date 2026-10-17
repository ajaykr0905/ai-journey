from dataclasses import replace
import unittest

from ai_journey.sft_data import ChatMessage, Conversation, SFTDataError


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
        self.assertEqual([row.role for row in conversation.messages], ["user", "assistant"])

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
            with self.subTest(role=role, content=repr(content)), self.assertRaises(
                SFTDataError
            ):
                ChatMessage(role, content)

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
