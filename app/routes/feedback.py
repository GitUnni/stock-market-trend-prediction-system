from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, Field
from jose import jwt, JWTError
from datetime import datetime, timezone

from app.deps import get_db
from app.auth import SECRET_KEY, ALGORITHM
from app.models import Feedback, User

router = APIRouter(prefix="/feedback", tags=["feedback"])

# ── Auth dependency (mirrors broadcasts.py) ──────────────────────────────────
_bearer = HTTPBearer(auto_error=False)


def get_current_user_payload(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """Decode the Bearer JWT and return the full payload."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated — Bearer token missing",
        )
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )
        return payload
    except (JWTError, ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


# ── Schemas ───────────────────────────────────────────────────────────────────

class FeedbackCreate(BaseModel):
    feedback_type: str = Field(..., pattern="^(REVIEW|COMPLAINT)$")
    rating: Optional[int] = Field(None, ge=1, le=5)   # REVIEW only
    subject: Optional[str] = None                      # COMPLAINT only
    content: str


class AdminReply(BaseModel):
    reply: str


class FeedbackOut(BaseModel):
    id: int
    user_id: int
    user_name: str
    user_email: str
    feedback_type: str
    rating: Optional[int]
    subject: Optional[str]
    content: str
    admin_reply: Optional[str]
    created_at: str
    replied_at: Optional[str]

    class Config:
        from_attributes = True


def _to_utc_iso(dt):
    """
    Return a UTC ISO-8601 string that always carries a timezone offset
    (e.g. '2025-05-01T10:30:00+00:00').

    SQLAlchemy may return a naive datetime even when the column is declared
    DateTime(timezone=True) — this happens with SQLite and some PostgreSQL
    driver configs. Tagging such values as UTC is safe because
    server_default=func.now() / datetime.now(timezone.utc) always writes UTC.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()   # e.g. "2025-05-01T10:30:00+00:00"


def _serialize(fb: Feedback) -> FeedbackOut:
    return FeedbackOut(
        id=fb.id,
        user_id=fb.user_id,
        user_name=fb.user_name,
        user_email=fb.user_email,
        feedback_type=fb.feedback_type,
        rating=fb.rating,
        subject=fb.subject,
        content=fb.content,
        admin_reply=fb.admin_reply,
        created_at=_to_utc_iso(fb.created_at) or "",
        replied_at=_to_utc_iso(fb.replied_at),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED)
def submit_feedback(
    payload: FeedbackCreate,
    db: Session = Depends(get_db),
    token_payload: dict = Depends(get_current_user_payload),
):
    """Customer submits a review or complaint."""
    role = token_payload.get("role")
    if role not in ("CUSTOMER", "INSTITUTION"):
        raise HTTPException(status_code=403, detail="Only customers can submit feedback.")

    # Validate: REVIEW needs rating, COMPLAINT needs subject
    if payload.feedback_type == "REVIEW" and payload.rating is None:
        raise HTTPException(status_code=422, detail="A rating is required for reviews.")
    if payload.feedback_type == "COMPLAINT" and not payload.subject:
        raise HTTPException(status_code=422, detail="A subject is required for complaints.")

    user_id = token_payload.get("user_id") or token_payload.get("sub")
    user_name  = token_payload.get("name", "Unknown")
    user_email = token_payload.get("email", "")

    # Resolve numeric user_id from DB if JWT only has email as sub
    if not str(user_id).isdigit():
        user = db.query(User).filter(User.email == user_email).first()
        user_id = user.id if user else 0

    fb = Feedback(
        user_id=int(user_id),
        user_name=user_name,
        user_email=user_email,
        feedback_type=payload.feedback_type,
        rating=payload.rating,
        subject=payload.subject.strip() if payload.subject else None,
        content=payload.content.strip(),
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)
    return {"message": "Feedback submitted successfully.", "id": fb.id}


@router.get("/my", response_model=List[FeedbackOut])
def get_my_feedback(
    db: Session = Depends(get_db),
    token_payload: dict = Depends(get_current_user_payload),
):
    """Customer retrieves their own submitted feedback (newest first)."""
    user_email = token_payload.get("email", "")
    user = db.query(User).filter(User.email == user_email).first()
    if not user:
        return []

    rows = (
        db.query(Feedback)
        .filter(Feedback.user_id == user.id)
        .order_by(Feedback.created_at.desc())
        .all()
    )
    return [_serialize(fb) for fb in rows]


@router.get("", response_model=List[FeedbackOut])
def get_all_feedback(
    db: Session = Depends(get_db),
    token_payload: dict = Depends(get_current_user_payload),
):
    """Admin retrieves all feedback (newest first)."""
    if token_payload.get("role") != "ADMIN":
        raise HTTPException(status_code=403, detail="Admin access required.")

    rows = (
        db.query(Feedback)
        .order_by(Feedback.created_at.desc())
        .all()
    )
    return [_serialize(fb) for fb in rows]


@router.put("/{feedback_id}/reply", status_code=status.HTTP_200_OK)
def reply_to_feedback(
    feedback_id: int,
    payload: AdminReply,
    db: Session = Depends(get_db),
    token_payload: dict = Depends(get_current_user_payload),
):
    """Admin replies to a specific feedback entry."""
    if token_payload.get("role") != "ADMIN":
        raise HTTPException(status_code=403, detail="Admin access required.")

    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb:
        raise HTTPException(status_code=404, detail="Feedback not found.")

    fb.admin_reply = payload.reply.strip()
    fb.replied_at  = datetime.now(timezone.utc)
    db.commit()
    db.refresh(fb)
    return {"message": "Reply sent successfully.", "id": fb.id}