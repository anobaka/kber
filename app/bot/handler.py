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

from app.config import config
from app.db.models import ChatMessage
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
            resp = self.client.request.request(
                lark.RawRequest.builder()
                .http_method("GET")
                .uri("/open-apis/bot/v3/info")
                .build()
            )
            if resp.success():
                import json as _json
                data = _json.loads(resp.raw.content)
                open_id = data.get("bot", {}).get("open_id", "")
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
                # Match by open_id if available, otherwise fall back to
                # checking if any mention exists (legacy behaviour).
                mention_id = getattr(getattr(m, "id", None), "open_id", None) or getattr(m, "id", None)
                if self._bot_open_id and mention_id == self._bot_open_id:
                    is_at_bot = True
                elif not self._bot_open_id:
                    # Fallback: if we couldn't fetch bot open_id, treat
                    # any mention as bot mention (same as before).
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
        """Build a minimal card JSON for progress notifications."""
        card = {
            "config": {"wide_screen_mode": True},
            "elements": [
                {"tag": "markdown", "content": text},
            ],
        }
        return json.dumps(card)

    def send_card(self, chat_id: str, title: str, content: str) -> None:
        """Send an interactive card message."""
        card = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "blue",
            },
            "elements": [
                {"tag": "markdown", "content": content},
            ],
        }
        self.send_message(chat_id, json.dumps(card), msg_type="interactive")

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
