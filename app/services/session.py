import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid

from app.config import settings

logger = logging.getLogger(__name__)


class SessionManager:
    """In-memory session management for conversation history"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_ttl = settings.session_ttl
        self.max_memory = settings.max_session_memory
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session or return existing one"""
        if session_id and session_id in self.sessions:
            # Update last accessed time
            self.sessions[session_id]["last_accessed"] = datetime.now()
            return session_id
        
        # Create new session
        new_session_id = session_id or str(uuid.uuid4())
        self.sessions[new_session_id] = {
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "messages": [],
            "metadata": {}
        }
        
        logger.info(f"Created new session: {new_session_id}")
        return new_session_id
    
    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a message to the session history"""
        try:
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found, creating new session")
                self.create_session(session_id)
            
            session = self.sessions[session_id]
            
            message = {
                "role": role,  # "user" or "assistant"
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            session["messages"].append(message)
            session["last_accessed"] = datetime.now()
            
            # Trim messages if exceeding max memory
            if len(session["messages"]) > self.max_memory:
                # Keep the first message (system prompt if any) and trim from oldest
                system_messages = [msg for msg in session["messages"] if msg["role"] == "system"]
                other_messages = [msg for msg in session["messages"] if msg["role"] != "system"]
                
                # Keep recent messages within limit
                keep_count = self.max_memory - len(system_messages)
                if keep_count > 0:
                    session["messages"] = system_messages + other_messages[-keep_count:]
                else:
                    session["messages"] = system_messages[-self.max_memory:]
                
                logger.info(f"Trimmed session {session_id} to {len(session['messages'])} messages")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {e}")
            return False
    
    def get_conversation_history(
        self, 
        session_id: str, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found")
            return []
        
        session = self.sessions[session_id]
        session["last_accessed"] = datetime.now()
        
        messages = session["messages"]
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session"""
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Cleared session: {session_id}")
                return True
            else:
                logger.warning(f"Session {session_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error clearing session {session_id}: {e}")
            return False
    
    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """Get metadata for a session"""
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        return {
            "session_id": session_id,
            "created_at": session["created_at"].isoformat(),
            "last_accessed": session["last_accessed"].isoformat(),
            "message_count": len(session["messages"]),
            "metadata": session["metadata"]
        }
    
    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions and return count of removed sessions"""
        try:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session_data in self.sessions.items():
                last_accessed = session_data["last_accessed"]
                if current_time - last_accessed > timedelta(seconds=self.session_ttl):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
            return len(expired_sessions)
            
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            return 0
    
    def get_all_sessions(self) -> Dict[str, Any]:
        """Get information about all active sessions"""
        return {
            "total_sessions": len(self.sessions),
            "sessions": {
                session_id: self.get_session_metadata(session_id)
                for session_id in self.sessions.keys()
            }
        }
    
    def update_session_metadata(
        self, 
        session_id: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        """Update session metadata"""
        try:
            if session_id not in self.sessions:
                logger.warning(f"Session {session_id} not found")
                return False
            
            self.sessions[session_id]["metadata"].update(metadata)
            self.sessions[session_id]["last_accessed"] = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating session metadata for {session_id}: {e}")
            return False


# Global session manager instance
session_manager = SessionManager() 