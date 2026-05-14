from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from app import models, schemas
from app.auth import hash_password, verify_password, create_access_token, SECRET_KEY, ALGORITHM
from app.deps import get_db
import os
from jose import jwt, JWTError
from typing import Optional
import json
from upstash_redis import Redis
from dotenv import load_dotenv

load_dotenv("app/.env")

router = APIRouter(prefix="/auth", tags=["Auth"])

# Redis-backed verification code storage (Upstash REST) with in-memory fallback for local dev
verification_codes = {}
password_reset_codes = {}

VERIFICATION_CODE_TTL_SECONDS = int(os.getenv("VERIFICATION_CODE_TTL_SECONDS", "600"))
MAX_VERIFICATION_ATTEMPTS = int(os.getenv("MAX_VERIFICATION_ATTEMPTS", "5"))

redis_url = os.getenv("UPSTASH_REDIS_REST_URL")
redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
redis_client: Optional[Redis] = None
if redis_url and redis_token:
    redis_client = Redis(url=redis_url, token=redis_token)


def _require_redis_for_codes():
    if redis_client is None:
        raise HTTPException(
            status_code=503,
            detail="Verification/reset code storage requires Redis in this environment.",
        )


def _verification_key(email: str, purpose: str) -> str:
    return f"{purpose}:{email.lower()}"


def _store_code(email: str, code: str, purpose: str):
    payload = json.dumps({"code": code, "attempts": 0})
    if redis_client:
        redis_client.setex(_verification_key(email, purpose), VERIFICATION_CODE_TTL_SECONDS, payload)
    else:
        _require_redis_for_codes()


def _get_code_data(email: str, purpose: str):
    if redis_client:
        payload = redis_client.get(_verification_key(email, purpose))
        if not payload:
            return None
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        return json.loads(payload)

    _require_redis_for_codes()
    return None


def _delete_code(email: str, purpose: str):
    if redis_client:
        redis_client.delete(_verification_key(email, purpose))
        return

    _require_redis_for_codes()


def _update_code_data(email: str, data: dict, purpose: str):
    if redis_client:
        ttl_remaining = redis_client.ttl(_verification_key(email, purpose))
        ttl = ttl_remaining if isinstance(ttl_remaining, int) and ttl_remaining > 0 else VERIFICATION_CODE_TTL_SECONDS
        redis_client.setex(_verification_key(email, purpose), ttl, json.dumps(data))
        return

    _require_redis_for_codes()

# -- Auth dependency for admin endpoints --
_bearer = HTTPBearer(auto_error=False)


def get_current_user_payload(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> dict:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated — Bearer token missing",
        )
    try:
        return jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except (JWTError, ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )


def require_admin(token_payload: dict = Depends(get_current_user_payload)) -> dict:
    if token_payload.get("role") != "ADMIN":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return token_payload


def generate_verification_code():
    """Generate a 6-digit verification code"""
    return str(secrets.randbelow(900000) + 100000)


def send_verification_email(email: str, code: str):
    """Send verification email with code"""
    # Email configuration - update these with your SMTP settings
    SMTP_SERVER = "smtp.gmail.com"  # Change to your SMTP server
    SMTP_PORT = 587
    SENDER_EMAIL = os.getenv("sender_email") # Email Address which sends the code
    SENDER_PASSWORD = os.getenv("sender_password")  # My Pass Password (Google)
    
    subject = "Email Verification Code - Stock Market System"
    body = f"""
    Hello,
    
    Your verification code is: {code}
    
    Please enter this code to verify your email address.
    This code will expire in 10 minutes.
    
    If you did not request this code, please ignore this email.
    
    Best regards,
    Stock Market Trading System Team
    """
    
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False


@router.post("/register/customer")
def register_customer(user: schemas.CustomerRegister, db: Session = Depends(get_db)):
    """Register a new customer"""
    existing = db.query(models.User).filter(models.User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = models.User(
        name=user.name,
        email=user.email,
        hashed_password=hash_password(user.password),
        contact_phone=user.contact_phone,
        role="CUSTOMER",
        status="PENDING_EMAIL_VERIFICATION",
        is_email_verified=False
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "Registration successful", "redirect": "/login"}


@router.post("/register/institution")
def register_institution(user: schemas.InstitutionRegister, db: Session = Depends(get_db)):
    """Register a new financial institution"""
    existing = db.query(models.User).filter(models.User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = models.User(
        name=user.name,
        email=user.email,
        hashed_password=hash_password(user.password),
        institution_name=user.institution_name,
        registration_number=user.registration_number,
        country=user.country,
        contact_person=user.contact_person,
        contact_phone=user.contact_phone,
        role="INSTITUTION",
        status="PENDING_EMAIL_VERIFICATION",
        is_email_verified=False
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "Registration successful", "redirect": "/login"}


@router.post("/login", response_model=schemas.TokenResponse)
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()

    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(
        data={
            "sub": str(db_user.id), 
            "role": db_user.role, 
            "email_verified": db_user.is_email_verified,
            "status": db_user.status,
            "name": db_user.name,
            "email": db_user.email
        }
    )

    return {"access_token": token, "token_type": "bearer"}


@router.post("/request-verification-code")
def request_verification_code(email: str, db: Session = Depends(get_db)):
    """Request a verification code for email verification"""
    user = db.query(models.User).filter(models.User.email == email).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.is_email_verified:
        raise HTTPException(status_code=400, detail="Email already verified")
    
    # Generate verification code
    code = generate_verification_code()
    
    _store_code(email, code, "email_verification")
    
    # Send email
    email_sent = send_verification_email(email, code)
    
    if not email_sent:
        raise HTTPException(status_code=502, detail="Unable to send verification email at the moment. Please try again.")
    
    return {"message": "Verification code sent to your email"}


@router.post("/verify-email")
def verify_email(email: str, code: str, db: Session = Depends(get_db)):
    """Verify email with the provided code"""
    user = db.query(models.User).filter(models.User.email == email).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.is_email_verified:
        raise HTTPException(status_code=400, detail="Email already verified")
    
    stored_data = _get_code_data(email, "email_verification")

    if not stored_data:
        raise HTTPException(status_code=400, detail="No verification code found or it has expired. Please request a new one.")

    if stored_data.get("attempts", 0) >= MAX_VERIFICATION_ATTEMPTS:
        _delete_code(email, "email_verification")
        raise HTTPException(status_code=429, detail="Too many attempts. Please request a new verification code.")

    if stored_data.get("code") != code:
        stored_data["attempts"] = stored_data.get("attempts", 0) + 1
        _update_code_data(email, stored_data, "email_verification")

        if stored_data["attempts"] >= MAX_VERIFICATION_ATTEMPTS:
            _delete_code(email, "email_verification")
            raise HTTPException(status_code=429, detail="Too many attempts. Please request a new verification code.")

        remaining_attempts = MAX_VERIFICATION_ATTEMPTS - stored_data["attempts"]
        raise HTTPException(status_code=400, detail=f"Invalid verification code. {remaining_attempts} attempt(s) remaining.")
    
    # Update user's email verification status
    user.is_email_verified = True
    
    # Set status based on role
    if user.role == "CUSTOMER":
        # Customer gets ACTIVE status immediately after email verification
        user.status = "ACTIVE"
    elif user.role == "INSTITUTION":
        # Institution needs admin approval, so set to PENDING_ADMIN_APPROVAL
        user.status = "PENDING_ADMIN_APPROVAL"
    
    db.commit()
    
    # Remove the used code
    _delete_code(email, "email_verification")
    
    # Generate new token with updated email_verified status
    token = create_access_token(
        data={
            "sub": str(user.id), 
            "role": user.role, 
            "email_verified": True,
            "status": user.status,
            "name": user.name,
            "email": user.email
        }
    )
    
    return {
        "message": "Email verified successfully",
        "access_token": token,
        "token_type": "bearer"
    }


@router.get("/admin/pending-institutions")
def get_pending_institutions(
    db: Session = Depends(get_db),
    _: dict = Depends(require_admin),
):
    """Get all institutions pending admin approval"""
    pending_institutions = db.query(models.User).filter(
        models.User.role == "INSTITUTION",
        models.User.status == "PENDING_ADMIN_APPROVAL",
        models.User.is_email_verified == True
    ).all()
    
    # Convert to list of dictionaries
    institutions_list = []
    for institution in pending_institutions:
        institutions_list.append({
            "id": institution.id,
            "name": institution.name,
            "email": institution.email,
            "institution_name": institution.institution_name,
            "registration_number": institution.registration_number,
            "country": institution.country,
            "contact_person": institution.contact_person,
            "contact_phone": institution.contact_phone,
            "created_at": institution.created_at.isoformat() if institution.created_at else None
        })
    
    return {"institutions": institutions_list}


@router.post("/admin/approve-institution/{institution_id}")
def approve_institution(
    institution_id: int,
    db: Session = Depends(get_db),
    _: dict = Depends(require_admin),
):
    """Approve a financial institution"""
    institution = db.query(models.User).filter(
        models.User.id == institution_id,
        models.User.role == "INSTITUTION"
    ).first()
    
    if not institution:
        raise HTTPException(status_code=404, detail="Institution not found")
    
    if institution.status == "ACTIVE":
        raise HTTPException(status_code=400, detail="Institution already approved")
    
    # Update status to ACTIVE
    institution.status = "ACTIVE"
    db.commit()
    
    return {"message": "Institution approved successfully"}


@router.post("/admin/reject-institution/{institution_id}")
def reject_institution(
    institution_id: int,
    db: Session = Depends(get_db),
    _: dict = Depends(require_admin),
):
    """Reject a financial institution"""
    institution = db.query(models.User).filter(
        models.User.id == institution_id,
        models.User.role == "INSTITUTION"
    ).first()
    
    if not institution:
        raise HTTPException(status_code=404, detail="Institution not found")
    
    # Update status to REJECTED
    institution.status = "REJECTED"
    db.commit()
    
    return {"message": "Institution rejected"}


@router.get("/admin/users")
def get_all_users(
    db: Session = Depends(get_db),
    _: dict = Depends(require_admin),
):
    """Get all registered users"""
    users = db.query(models.User).all()

    return {
        "users": [
            {
                "id":            u.id,
                "name":          u.name,
                "email":         u.email,
                "role":          u.role,
                "contact_phone": u.contact_phone or "",
            }
            for u in users
        ]
    }


@router.delete("/admin/users/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    _: dict = Depends(require_admin),
):
    """Delete a user by ID"""
    user = db.query(models.User).filter(models.User.id == user_id).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()

    return {"message": f"User '{user.name}' deleted successfully"}


# Forgot Password Endpoints


@router.post("/request-password-reset")
def request_password_reset(email: str, db: Session = Depends(get_db)):
    """Request a password reset code"""
    user = db.query(models.User).filter(models.User.email == email).first()

    if not user:
        raise HTTPException(status_code=404, detail="Email not registered")

    # Generate reset code
    code = generate_verification_code()

    _store_code(email, code, "password_reset")

    # Send email
    subject = "Password Reset Code - Stock Market System"
    body = f"""
    Hello {user.name},

    Your password reset code is: {code}

    Please enter this code to reset your password.
    This code will expire in 10 minutes.

    If you did not request this code, please ignore this email and your password will remain unchanged.

    Best regards,
    Stock Market Trading System Team
    """

    try:
        email_sent = send_verification_email(email, code)

        if not email_sent:
            # For development, return the code
            print(f"Password reset code for {email}: {code}")
            return {"message": "Reset code generated (check console in development)", "dev_code": code}

        return {"message": "Password reset code sent to your email"}
    except Exception as e:
        # For development, return the code
        print(f"Password reset code for {email}: {code}")
        return {"message": "Reset code generated (check console in development)", "dev_code": code}


@router.post("/verify-reset-code")
def verify_reset_code(email: str, code: str, db: Session = Depends(get_db)):
    """Verify the password reset code"""
    user = db.query(models.User).filter(models.User.email == email).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    stored_data = _get_code_data(email, "password_reset")

    if not stored_data:
        raise HTTPException(status_code=400, detail="No reset code found or it has expired. Please request a new one.")

    if stored_data.get("attempts", 0) >= MAX_VERIFICATION_ATTEMPTS:
        _delete_code(email, "password_reset")
        raise HTTPException(status_code=429, detail="Too many attempts. Please request a new reset code.")

    if stored_data.get("code") != code:
        stored_data["attempts"] = stored_data.get("attempts", 0) + 1
        _update_code_data(email, stored_data, "password_reset")

        if stored_data["attempts"] >= MAX_VERIFICATION_ATTEMPTS:
            _delete_code(email, "password_reset")
            raise HTTPException(status_code=429, detail="Too many attempts. Please request a new reset code.")

        remaining_attempts = MAX_VERIFICATION_ATTEMPTS - stored_data["attempts"]
        raise HTTPException(status_code=400, detail=f"Invalid reset code. {remaining_attempts} attempt(s) remaining.")

    return {"message": "Code verified successfully"}


@router.post("/reset-password")
def reset_password(
    reset_data: schemas.PasswordReset,
    db: Session = Depends(get_db)
):
    """Reset password with verified code"""
    user = db.query(models.User).filter(models.User.email == reset_data.email).first()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Verify code one more time
    stored_data = _get_code_data(reset_data.email, "password_reset")

    if not stored_data or stored_data.get("code") != reset_data.code:
        raise HTTPException(status_code=400, detail="Invalid or expired reset code")

    # Update password
    user.hashed_password = hash_password(reset_data.new_password)
    db.commit()

    # Remove the used code
    _delete_code(reset_data.email, "password_reset")

    return {"message": "Password reset successfully"}