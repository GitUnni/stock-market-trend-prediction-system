from pydantic import BaseModel, EmailStr, constr

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

# ---------- Customer ----------
class CustomerRegister(BaseModel):
    name: str
    email: EmailStr
    password: constr(min_length=8, max_length=64)
    contact_phone: str

# ---------- Institution ----------
class InstitutionRegister(BaseModel):
    name: str
    email: EmailStr
    password: constr(min_length=8, max_length=64)
    institution_name: str
    registration_number: str
    country: str
    contact_person: str
    contact_phone: str

class PasswordReset(BaseModel):
    email: str
    code: str
    new_password: str