from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from jose import jwt, JWTError
from app.deps import get_db
from app.auth import SECRET_KEY, ALGORITHM
from app.models import Broadcast

router = APIRouter(prefix="/broadcasts", tags=["broadcasts"])

# -- Auth dependency --
_bearer = HTTPBearer(auto_error=False)


def get_current_user_payload(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    """Decode the Bearer JWT and return the full payload (includes role, sub, etc.)."""
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


# -- Schemas --

class BroadcastCreate(BaseModel):
    title: str
    content: str


class BroadcastOut(BaseModel):
    id: int
    title: str
    content: str
    created_at: str

    class Config:
        from_attributes = True


# -- Endpoints --

@router.post("", status_code=status.HTTP_201_CREATED)
def create_broadcast(
    payload: BroadcastCreate,
    db: Session = Depends(get_db),
    token_payload: dict = Depends(get_current_user_payload),
):
    """Admin-only: store a new broadcast message."""
    if token_payload.get("role") != "ADMIN":
        raise HTTPException(status_code=403, detail="Admin access required.")

    broadcast = Broadcast(title=payload.title.strip(), content=payload.content.strip())
    db.add(broadcast)
    db.commit()
    db.refresh(broadcast)
    return {"message": "Broadcast sent successfully.", "id": broadcast.id}


@router.get("", response_model=List[BroadcastOut])
def get_broadcasts(
    db: Session = Depends(get_db),
    token_payload: dict = Depends(get_current_user_payload),
):
    """Return all broadcasts newest-first (accessible to all authenticated users)."""
    broadcasts = (
        db.query(Broadcast)
        .order_by(Broadcast.created_at.desc())
        .all()
    )
    return [
        BroadcastOut(
            id=b.id,
            title=b.title,
            content=b.content,
            created_at=b.created_at.isoformat() if b.created_at else "",
        )
        for b in broadcasts
    ]