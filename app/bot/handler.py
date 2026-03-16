"""Feishu bot event handler – receives messages via WebSocket long connection."""

import json
import logging
from typing import Any

import lark_oapi as lark
from lark_oapi.api.im.v1 import (
    CreateMessageRequest,
    CreateMessageRequestBody,
    GetChatRequest,
    ListMessageRequest,
)

from app.config import config
from app.db.models import ChatMessage
from app.db.session import get_session

logger = logging.getLogger(__name__)


class FeishuBot:
    """Feishu (Lark) bot for receiving and sending messages."""

    def __init__(self) -> None:
        self._client: lark.Client | None = None
        self._ws_client: Any = None

    @property
    def client(self) -> lark.Client:
        if self._client is None:
            self._client = lark.Client.builder() \
                .app_id(config.FEISHU_APP_ID) \
                .app_secret(config.FEISHU_APP_SECRET) \
                .log_level(lark.LogLevel.WARNING) \
                .build()
        return self._client

    def start(self, message_handler: Any) -> None:
        """Start the WebSocket long connection to receive messages."""
        from app.bot.commands import CommandRouter

        router = CommandRouter(self)

        event_handler = lark.EventDispatcherHandler.builder("", "") \
            .register_p2_im_message_receive_v1(
                lambda ctx, event: self._on_message(ctx, event, router)
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

    def _on_message(self, ctx: Any, event: Any, router: Any) -> None:
        """Handle incoming message event."""
        try:
            msg = event.event.message
            chat_id = msg.chat_id
            message_id = msg.message_id
            msg_type = msg.message_type
            sender_id = event.event.sender.sender_id.open_id

            # Parse message content
            content_str = msg.content
            content_json = json.loads(content_str) if content_str else {}
            text = content_json.get("text", "").strip()

            # Check if bot is mentioned (for group chats)
            mentions = msg.mentions or []
            is_at_bot = any(
                m.key in text for m in mentions if m.id and m.id.app_id == config.FEISHU_APP_ID
            ) if mentions else False

            # Remove @bot mention from text
            for m in mentions:
                if m.key:
                    text = text.replace(m.key, "").strip()

            parent_id = getattr(msg, "parent_id", None)

            if is_at_bot or msg.chat_type == "p2p":
                # This is a command or question directed at the bot
                router.handle(chat_id, message_id, sender_id, text)
            else:
                # This is a regular group message – collect for knowledge base
                self._collect_message(chat_id, message_id, sender_id, text, msg_type, parent_id)

        except Exception:
            logger.exception("Error handling message event")

    def _collect_message(
        self,
        chat_id: str,
        message_id: str,
        sender_id: str,
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
                    content=content,
                    msg_type=msg_type,
                ))
        except Exception:
            logger.exception("Failed to collect message %s", message_id)

    def send_message(self, chat_id: str, text: str, msg_type: str = "text") -> None:
        """Send a message to a Feishu chat."""
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
        except Exception:
            logger.exception("Failed to send message to %s", chat_id)

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
                        messages.append({
                            "message_id": item.message_id,
                            "chat_id": chat_id,
                            "sender_id": item.sender.id if item.sender else None,
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
