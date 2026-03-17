"""Feishu bot event handler – receives messages via WebSocket long connection."""

import collections
import json
import logging
import threading
from typing import Any

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    GetChatRequest,
    ListMessageRequest,
    PatchMessageRequest,
    PatchMessageRequestBody,
)
from lark_oapi.event.callback.model.p2_card_action_trigger import (
    P2CardActionTrigger,
    P2CardActionTriggerResponse,
)

from app.config import config
from app.db.models import ChatMessage, ResponseFeedback
from app.db.session import get_session

logger = logging.getLogger(__name__)


_DEDUP_MAX = 1024


class FeishuBot:
    """Feishu (Lark) bot for receiving and sending messages."""

    def __init__(self) -> None:
        self._client: lark.Client | None = None
        self._ws_client: Any = None
        self._bot_open_id: str | None = None
        self._seen_msg_ids: collections.OrderedDict[str, None] = collections.OrderedDict()
        self._seen_lock = threading.Lock()

    @property
    def client(self) -> lark.Client:
        if self._client is None:
            self._client = lark.Client.builder() \
                .app_id(config.FEISHU_APP_ID) \
                .app_secret(config.FEISHU_APP_SECRET) \
                .log_level(lark.LogLevel.WARNING) \
                .build()
        return self._client

    def _fetch_bot_open_id(self) -> str | None:
        """Fetch the bot's own open_id via Feishu API (bot info endpoint)."""
        try:
            req = lark.BaseRequest()
            req.http_method = lark.HttpMethod.GET
            req.uri = "/open-apis/bot/v3/info"
            req.token_types = {lark.AccessTokenType.TENANT}
            resp = self.client.request(req)
            logger.info(
                "Bot info API response: success=%s, status_code=%s, raw=%s",
                resp.success(),
                getattr(getattr(resp, "raw", None), "status_code", None),
                getattr(getattr(resp, "raw", None), "content", None),
            )
            if resp.success():
                import json as _json
                data = _json.loads(resp.raw.content)
                # The response might be {"bot": {"open_id": ...}} or {"data": {"open_id": ...}}
                open_id = (
                    data.get("bot", {}).get("open_id", "")
                    or data.get("data", {}).get("open_id", "")
                )
                if open_id:
                    logger.info("Bot open_id: %s", open_id)
                    return open_id
            logger.warning("Failed to fetch bot info: %s", getattr(resp, "msg", ""))
        except Exception:
            logger.exception("Failed to fetch bot open_id")
        return None

    def start(self, message_handler: Any) -> None:
        """Start the WebSocket long connection to receive messages."""
        from app.bot.commands import CommandRouter
        from app.services.debug_notifier import set_send_fn, set_send_progress_fn, set_update_fn

        # Fetch bot's own open_id for @mention detection
        self._bot_open_id = self._fetch_bot_open_id()

        # Register send/update functions for debug notifications
        set_send_fn(self.send_message)
        set_update_fn(self.update_message)
        set_send_progress_fn(self.send_progress_card)

        router = CommandRouter(self)

        event_handler = lark.EventDispatcherHandler.builder("", "") \
            .register_p2_im_message_receive_v1(
                lambda event: self._on_message(None, event, router)
            ) \
            .register_p2_card_action_trigger(self._on_card_action) \
            .build()

        self._ws_client = lark.ws.Client(
            config.FEISHU_APP_ID,
            config.FEISHU_APP_SECRET,
            event_handler=event_handler,
            log_level=lark.LogLevel.WARNING,
        )
        logger.info("Starting Feishu bot WebSocket connection...")
        self._ws_client.start()

    def _is_duplicate(self, message_id: str) -> bool:
        """Return True if this message_id was already seen (LRU dedup)."""
        with self._seen_lock:
            if message_id in self._seen_msg_ids:
                return True
            self._seen_msg_ids[message_id] = None
            if len(self._seen_msg_ids) > _DEDUP_MAX:
                self._seen_msg_ids.popitem(last=False)
            return False

    def _on_message(self, ctx: Any, event: Any, router: Any) -> None:
        """Handle incoming message event."""
        try:
            msg = event.event.message
            chat_id = msg.chat_id
            message_id = msg.message_id

            if self._is_duplicate(message_id):
                logger.debug("Duplicate message ignored: %s", message_id)
                return

            # Log full sender structure for debugging
            sender = event.event.sender
            sender_id_obj = sender.sender_id if sender else None
            logger.debug(
                "Message %s sender structure: sender_id=%s, sender_type=%s, "
                "open_id=%s, user_id=%s, union_id=%s",
                message_id,
                sender_id_obj,
                getattr(sender, "sender_type", None),
                getattr(sender_id_obj, "open_id", None),
                getattr(sender_id_obj, "user_id", None),
                getattr(sender_id_obj, "union_id", None),
            )

            msg_type = msg.message_type
            sender_id = sender_id_obj.open_id if sender_id_obj else ""
            user_id = getattr(sender_id_obj, "user_id", None) or ""

            # Parse message content
            content_str = msg.content
            content_json = json.loads(content_str) if content_str else {}
            text = content_json.get("text", "").strip()

            # Check if bot is specifically mentioned (for group chats)
            mentions = msg.mentions or []
            is_at_bot = False
            for m in mentions:
                mention_id_obj = getattr(m, "id", None)
                mention_open_id = getattr(mention_id_obj, "open_id", None)
                # If m.id is a UserId object, use its open_id; otherwise m.id itself might be a string
                mention_id = mention_open_id or mention_id_obj
                logger.debug(
                    "Mention: key=%s, name=%s, id_obj=%s, open_id=%s, bot_open_id=%s",
                    getattr(m, "key", None), getattr(m, "name", None),
                    mention_id_obj, mention_open_id, self._bot_open_id,
                )
                if self._bot_open_id and mention_id == self._bot_open_id:
                    is_at_bot = True
                elif not self._bot_open_id:
                    # Fallback: if we couldn't fetch bot open_id, treat
                    # any mention as bot mention (same as before).
                    logger.warning(
                        "bot_open_id not set, falling back to treating all mentions as bot"
                    )
                    is_at_bot = True

            # Remove @mention placeholders from text
            for m in mentions:
                if m.key:
                    text = text.replace(m.key, "").strip()

            parent_id = getattr(msg, "parent_id", None)

            if is_at_bot or msg.chat_type == "p2p":
                # This is a command or question directed at the bot
                router.handle(chat_id, message_id, sender_id, text, user_id=user_id)
            else:
                # Regular group message (including @others) – collect for knowledge base
                self._collect_message(chat_id, message_id, sender_id, user_id, text, msg_type, parent_id)

        except Exception:
            logger.exception("Error handling message event")

    def _collect_message(
        self,
        chat_id: str,
        message_id: str,
        sender_id: str,
        user_id: str,
        content: str,
        msg_type: str,
        parent_id: str | None,
    ) -> None:
        """Store a group message for later analysis."""
        if msg_type != "text" or not content:
            return

        from app.bot.commands import CommandRouter

        # Check if this chat has any KB bindings
        if not CommandRouter.chat_has_kb_binding(chat_id):
            return

        try:
            with get_session() as session:
                # Dedup by message_id
                from sqlalchemy import select
                exists = session.execute(
                    select(ChatMessage.id).where(ChatMessage.message_id == message_id)
                ).scalar_one_or_none()
                if exists:
                    return

                session.add(ChatMessage(
                    chat_id=chat_id,
                    message_id=message_id,
                    parent_id=parent_id,
                    sender_id=sender_id,
                    user_id=user_id or None,
                    content=content,
                    msg_type=msg_type,
                ))
        except Exception:
            logger.exception("Failed to collect message %s", message_id)

    def send_message(self, chat_id: str, text: str, msg_type: str = "text") -> str | None:
        """Send a message to a Feishu chat. Returns the message_id on success."""
        try:
            if msg_type == "text":
                content = json.dumps({"text": text})
            elif msg_type == "interactive":
                content = text  # Already JSON
            elif msg_type == "post":
                content = text  # Already JSON (post structure)
            else:
                content = json.dumps({"text": text})

            request = CreateMessageRequest.builder() \
                .receive_id_type("chat_id") \
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(chat_id)
                    .msg_type(msg_type)
                    .content(content)
                    .build()
                ) \
                .build()

            response = self.client.im.v1.message.create(request)
            if not response.success():
                logger.error("Failed to send message: %s", response.msg)
                return None
            return response.data.message_id if response.data else None
        except Exception:
            logger.exception("Failed to send message to %s", chat_id)
            return None

    def update_message(self, message_id: str, text: str) -> bool:
        """Update an existing Feishu card message in-place. Returns True on success.

        Note: Feishu only supports updating interactive (card) messages.
        """
        try:
            card_json = self._build_progress_card(text)
            request = PatchMessageRequest.builder() \
                .message_id(message_id) \
                .request_body(
                    PatchMessageRequestBody.builder()
                    .content(card_json)
                    .build()
                ) \
                .build()

            response = self.client.im.v1.message.patch(request)
            if not response.success():
                logger.error("Failed to update message %s: %s", message_id, response.msg)
                return False
            return True
        except Exception:
            logger.exception("Failed to update message %s", message_id)
            return False

    def send_progress_card(self, chat_id: str, text: str) -> str | None:
        """Send a progress card that can be updated in-place later. Returns message_id."""
        return self.send_message(chat_id, self._build_progress_card(text), msg_type="interactive")

    @staticmethod
    def _build_progress_card(text: str) -> str:
        """Build a minimal card JSON 2.0 for progress notifications."""
        card = {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "body": {
                "elements": [
                    {"tag": "markdown", "content": text, "text_size": "normal"},
                ],
            },
        }
        return json.dumps(card)

    def send_card(self, chat_id: str, title: str, content: str) -> str | None:
        """Send an interactive card with JSON 2.0 Markdown rendering.

        Returns the message_id on success.
        """
        card = {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "blue",
            },
            "body": {
                "elements": [
                    {"tag": "markdown", "content": content, "text_size": "normal"},
                ],
            },
        }
        return self.send_message(chat_id, json.dumps(card), msg_type="interactive")

    def send_rag_answer_card(
        self,
        chat_id: str,
        question: str,
        answer: str,
    ) -> str | None:
        """Send an interactive card with the RAG answer and feedback buttons.

        Returns the message_id of the sent card for feedback tracking.
        """
        card = self._build_rag_answer_card(answer, question)
        return self.send_message(chat_id, json.dumps(card), msg_type="interactive")

    @staticmethod
    def _build_rag_answer_card(
        answer: str,
        question: str,
        *,
        feedback_state: str | None = None,
        feedback_reason: str | None = None,
    ) -> dict:
        """Build the interactive card JSON 2.0 for a RAG answer.

        Args:
            answer: The RAG answer markdown content.
            question: The original question (truncated, stored in button value).
            feedback_state: None (initial), "helpful", "not_helpful_ask", or "not_helpful_done".
            feedback_reason: The reason text (when not_helpful_done).
        """
        elements: list[dict] = [
            {
                "tag": "markdown",
                "content": answer,
                "text_size": "normal",
                "margin": "0px 0px 8px 0px",
            },
            {"tag": "hr"},
        ]

        # Truncate question/answer for storage in button value (Feishu has value size limits).
        # Answer is stored in the value so the callback can rebuild the card
        # (P2CardActionTrigger does not include the original card content).
        q_short = question[:200] if question else ""
        a_short = answer[:1500] if answer else ""

        # Common callback value payload for buttons
        _cb_base = {"question": q_short, "answer": a_short}

        if feedback_state is None:
            # Initial state: two feedback buttons side by side via column_set
            elements.append({
                "tag": "column_set",
                "flex_mode": "flow",
                "columns": [
                    {
                        "tag": "column",
                        "width": "auto",
                        "weight": 1,
                        "elements": [{
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": "👍 有用"},
                            "type": "primary",
                            "behaviors": [{"type": "callback", "value": {**_cb_base, "action": "feedback_helpful"}}],
                        }],
                    },
                    {
                        "tag": "column",
                        "width": "auto",
                        "weight": 1,
                        "elements": [{
                            "tag": "button",
                            "text": {"tag": "plain_text", "content": "👎 没用"},
                            "type": "default",
                            "behaviors": [{"type": "callback", "value": {**_cb_base, "action": "feedback_not_helpful"}}],
                        }],
                    },
                ],
            })
        elif feedback_state == "helpful":
            elements.append({
                "tag": "markdown",
                "content": "👍 **感谢你的反馈！**",
                "text_size": "notation",
                "margin": "4px 0px 0px 0px",
            })
        elif feedback_state == "not_helpful_ask":
            # Show form for reason input + submit/skip buttons
            elements.append({
                "tag": "markdown",
                "content": "👎 感谢反馈！如果方便，请告诉我们哪里可以改进：",
                "text_size": "normal",
                "margin": "4px 0px 0px 0px",
            })
            elements.append({
                "tag": "form_container",
                "name": "feedback_form",
                "elements": [
                    {
                        "tag": "input",
                        "name": "feedback_reason",
                        "placeholder": {"tag": "plain_text", "content": "请输入原因（可选）"},
                        "width": "fill",
                    },
                    {
                        "tag": "column_set",
                        "flex_mode": "flow",
                        "columns": [
                            {
                                "tag": "column",
                                "width": "auto",
                                "weight": 1,
                                "elements": [{
                                    "tag": "button",
                                    "text": {"tag": "plain_text", "content": "提交"},
                                    "type": "primary",
                                    "name": "submit_btn",
                                    "form_action_type": "submit",
                                    "behaviors": [{"type": "callback", "value": {**_cb_base, "action": "feedback_submit_reason"}}],
                                }],
                            },
                            {
                                "tag": "column",
                                "width": "auto",
                                "weight": 1,
                                "elements": [{
                                    "tag": "button",
                                    "text": {"tag": "plain_text", "content": "跳过"},
                                    "type": "default",
                                    "name": "skip_btn",
                                    "form_action_type": "submit",
                                    "behaviors": [{"type": "callback", "value": {**_cb_base, "action": "feedback_skip_reason"}}],
                                }],
                            },
                        ],
                    },
                ],
            })
        elif feedback_state == "not_helpful_done":
            reason_text = ""
            if feedback_reason:
                reason_text = f"\n原因：{feedback_reason}"
            elements.append({
                "tag": "markdown",
                "content": f"👎 **感谢你的反馈，我们会持续改进！**{reason_text}",
                "text_size": "notation",
                "margin": "4px 0px 0px 0px",
            })

        card = {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": "知识库回答"},
                "template": "blue",
            },
            "body": {
                "elements": elements,
            },
        }
        return card

    def _on_card_action(self, data: P2CardActionTrigger) -> P2CardActionTriggerResponse:
        """SDK callback for card action events (registered via register_p2_card_action_trigger)."""
        card_data = self._handle_card_action(data)
        if card_data is None:
            return P2CardActionTriggerResponse({})
        return P2CardActionTriggerResponse({
            "card": {
                "type": "raw",
                "data": card_data,
            },
        })

    def _handle_card_action(self, data: P2CardActionTrigger) -> dict | None:
        """Process card action and return an updated card dict, or None."""
        try:
            event = data.event
            if not event:
                return None

            action = event.action
            if not action:
                return None

            action_value = action.value or {}
            action_type = action_value.get("action", "")
            question = action_value.get("question", "")

            # Operator info
            operator = event.operator
            user_open_id = operator.open_id if operator else ""

            # Message and chat context
            ctx = event.context
            open_message_id = ctx.open_message_id if ctx else ""
            chat_id = ctx.open_chat_id if ctx else ""

            # Extract the original answer from the card to preserve in the updated card.
            # The card content is not available in P2CardActionTrigger, so we look it up
            # from the feedback record or fall back to fetching via API.
            # For simplicity, we store the answer in the button value alongside the question.
            answer_content = action_value.get("answer", "")

            # Form input values (for reason submission)
            form_value = action.form_value or {}

            if action_type == "feedback_helpful":
                self._save_feedback(
                    chat_id=chat_id,
                    message_id=open_message_id,
                    user_open_id=user_open_id,
                    question=question,
                    answer=answer_content,
                    rating="helpful",
                )
                return self._build_rag_answer_card(
                    answer_content, question, feedback_state="helpful",
                )

            elif action_type == "feedback_not_helpful":
                self._save_feedback(
                    chat_id=chat_id,
                    message_id=open_message_id,
                    user_open_id=user_open_id,
                    question=question,
                    answer=answer_content,
                    rating="not_helpful",
                )
                return self._build_rag_answer_card(
                    answer_content, question, feedback_state="not_helpful_ask",
                )

            elif action_type == "feedback_submit_reason":
                reason = form_value.get("feedback_reason", "")
                self._update_feedback_reason(open_message_id, reason)
                return self._build_rag_answer_card(
                    answer_content, question,
                    feedback_state="not_helpful_done",
                    feedback_reason=reason,
                )

            elif action_type == "feedback_skip_reason":
                return self._build_rag_answer_card(
                    answer_content, question,
                    feedback_state="not_helpful_done",
                )

        except Exception:
            logger.exception("Error handling card action")
        return None

    @staticmethod
    def _save_feedback(
        *,
        chat_id: str,
        message_id: str,
        user_open_id: str,
        question: str,
        answer: str,
        rating: str,
        reason: str | None = None,
    ) -> None:
        """Save feedback to the database."""
        if not message_id:
            logger.warning("Cannot save feedback: no message_id")
            return
        try:
            with get_session() as session:
                from sqlalchemy import select as sa_select
                existing = session.execute(
                    sa_select(ResponseFeedback.id).where(
                        ResponseFeedback.message_id == message_id,
                    )
                ).scalar_one_or_none()
                if existing:
                    return  # Already recorded
                session.add(ResponseFeedback(
                    chat_id=chat_id,
                    message_id=message_id,
                    user_open_id=user_open_id or None,
                    question=question or None,
                    answer=answer or None,
                    rating=rating,
                    reason=reason,
                ))
        except Exception:
            logger.exception("Failed to save feedback for message %s", message_id)

    @staticmethod
    def _update_feedback_reason(message_id: str, reason: str) -> None:
        """Update the reason field for an existing feedback record."""
        if not message_id or not reason:
            return
        try:
            with get_session() as session:
                from sqlalchemy import update as sa_update
                session.execute(
                    sa_update(ResponseFeedback).where(
                        ResponseFeedback.message_id == message_id,
                    ).values(reason=reason)
                )
        except Exception:
            logger.exception("Failed to update feedback reason for message %s", message_id)

    def fetch_history_messages(self, chat_id: str, page_size: int = 50) -> list[dict[str, Any]]:
        """Fetch historical messages from a chat for compensation."""
        messages: list[dict[str, Any]] = []
        page_token: str | None = None

        try:
            while True:
                builder = ListMessageRequest.builder() \
                    .container_id_type("chat") \
                    .container_id(chat_id) \
                    .page_size(page_size)

                if page_token:
                    builder = builder.page_token(page_token)

                request = builder.build()
                response = self.client.im.v1.message.list(request)

                if not response.success():
                    logger.error("Failed to fetch history: %s", response.msg)
                    break

                items = response.data.items or []
                for item in items:
                    try:
                        content_json = json.loads(item.body.content) if item.body and item.body.content else {}
                        # Extract sender info – structure may vary by API version
                        item_sender = item.sender if item else None
                        item_sender_id = getattr(item_sender, "id", None) if item_sender else None
                        logger.debug(
                            "History item sender structure: %s", item_sender,
                        )
                        messages.append({
                            "message_id": item.message_id,
                            "chat_id": chat_id,
                            "sender_id": item_sender_id,
                            "content": content_json.get("text", ""),
                            "msg_type": item.msg_type,
                            "parent_id": item.parent_id,
                            "create_time": item.create_time,
                        })
                    except Exception:
                        continue

                page_token = response.data.page_token
                if not page_token or not items:
                    break

        except Exception:
            logger.exception("Failed to fetch history for %s", chat_id)

        return messages


feishu_bot = FeishuBot()
