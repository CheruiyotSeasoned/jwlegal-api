from sqlalchemy import Column, String, Text, DateTime, Boolean, ForeignKey, Enum, BigInteger, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
from app.models import ConversationType, MessageType, ParticipantRole
import uuid

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255))
    type = Column(Enum(ConversationType), default=ConversationType.DIRECT)
    case_id = Column(String(36), ForeignKey("cases.id"))
    created_by = Column(String(36), ForeignKey("users.id"), nullable=False)
    is_archived = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    case = relationship("Case", back_populates="conversations")
    creator = relationship("User", foreign_keys=[created_by])
    participants = relationship("ConversationParticipant", back_populates="conversation")
    messages = relationship("Message", back_populates="conversation")

class ConversationParticipant(Base):
    __tablename__ = "conversation_participants"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    role = Column(Enum(ParticipantRole), default=ParticipantRole.PARTICIPANT)
    joined_at = Column(DateTime(timezone=True), server_default=func.now())
    left_at = Column(DateTime(timezone=True))
    is_muted = Column(Boolean, default=False)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="participants")
    user = relationship("User")

class Message(Base):
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False)
    sender_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    message_type = Column(Enum(MessageType), default=MessageType.TEXT)
    file_url = Column(String(500))
    file_name = Column(String(255))
    file_size = Column(BigInteger)
    
    # Message status
    is_edited = Column(Boolean, default=False)
    edited_at = Column(DateTime(timezone=True))
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime(timezone=True))
    deleted_by = Column(String(36), ForeignKey("users.id"))
    
    # Message metadata
    reply_to_message_id = Column(String(36), ForeignKey("messages.id"))
    forwarded_from_message_id = Column(String(36), ForeignKey("messages.id"))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    sender = relationship("User", foreign_keys=[sender_id])
    deleter = relationship("User", foreign_keys=[deleted_by])
    reply_to_message = relationship("Message", remote_side=[id], foreign_keys=[reply_to_message_id])
    forwarded_from_message = relationship("Message", remote_side=[id], foreign_keys=[forwarded_from_message_id])
    read_receipts = relationship("MessageRead", back_populates="message")

class MessageRead(Base):
    __tablename__ = "message_reads"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    message_id = Column(String(36), ForeignKey("messages.id"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    read_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    message = relationship("Message", back_populates="read_receipts")
    user = relationship("User")


