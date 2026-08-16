import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.domain.entities.message_role import MessageRole
from app.infrastructure.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user")
    devices: Mapped[list["Device"]] = relationship(back_populates="user")


class Device(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "devices"

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))

    user: Mapped["User"] = relationship(back_populates="devices")


class SceneModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "scenes"

    image_storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    image_filename: Mapped[str] = mapped_column(String(255))
    image_mime_type: Mapped[str] = mapped_column(String(100))
    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)
    image_size_bytes: Mapped[int] = mapped_column(Integer)
    image_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    conversation: Mapped["Conversation | None"] = relationship(back_populates="scene")
    detected_objects: Mapped[list["DetectedObject"]] = relationship(
        back_populates="scene", cascade="all, delete-orphan"
    )


class Conversation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "conversations"
    __table_args__ = (UniqueConstraint("scene_id", name="uq_conversations_scene_id"),)

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)
    scene_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("scenes.id"), index=True)

    user: Mapped["User"] = relationship(back_populates="conversations")
    scene: Mapped["SceneModel"] = relationship(back_populates="conversation")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class DetectedObject(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "detected_objects"

    scene_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("scenes.id"), index=True)

    class_id: Mapped[int] = mapped_column(Integer)
    class_name: Mapped[str] = mapped_column(String(100))
    confidence: Mapped[float] = mapped_column(Float)

    bbox_x1: Mapped[int] = mapped_column(Integer)
    bbox_y1: Mapped[int] = mapped_column(Integer)
    bbox_x2: Mapped[int] = mapped_column(Integer)
    bbox_y2: Mapped[int] = mapped_column(Integer)

    position_horizontal: Mapped[str] = mapped_column(String(10))
    position_vertical: Mapped[str] = mapped_column(String(10))
    position_region: Mapped[str] = mapped_column(String(20))

    color_name: Mapped[str] = mapped_column(String(50))
    color_r: Mapped[int] = mapped_column(Integer)
    color_g: Mapped[int] = mapped_column(Integer)
    color_b: Mapped[int] = mapped_column(Integer)
    color_confidence: Mapped[float] = mapped_column(Float)

    scene: Mapped["SceneModel"] = relationship(back_populates="detected_objects")


class Message(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "messages"

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole, native_enum=False, length=20))
    content: Mapped[str] = mapped_column(Text)

    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")
