from app.database import SessionLocal
from app.models import User
from app.auth import hash_password

ADMIN_EMAIL = "sudoadmin@gmail.com"
ADMIN_NAME = "binny"
ADMIN_PASSWORD = "pseudoadmin"  

def create_admin():
    db = SessionLocal()

    existing = db.query(User).filter(User.email == ADMIN_EMAIL).first()
    if existing:
        print("❌ Admin already exists")
        return

    admin = User(
        name=ADMIN_NAME,
        email=ADMIN_EMAIL,
        hashed_password=hash_password(ADMIN_PASSWORD),
        role="ADMIN",
        status="ACTIVE",
        is_email_verified=True
    )

    db.add(admin)
    db.commit()
    db.close()

    print("✅ Admin user created successfully")

if __name__ == "__main__":
    create_admin()
