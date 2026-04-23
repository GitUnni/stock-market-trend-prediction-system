from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["Pages"])

templates = Jinja2Templates(directory="app/templates")


@router.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "home.html", {"request": request}
    )

@router.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request}
    )

@router.get("/register")
def register_choice(request: Request):
    return templates.TemplateResponse(
        "register_choice.html", {"request": request}
    )

@router.get("/register/customer")
def register_customer_page(request: Request):
    return templates.TemplateResponse(
        "register_customer.html",
        {"request": request}
    )

@router.get("/register/institution")
def register_institution_page(request: Request):
    return templates.TemplateResponse(
        "register_fin.html",
        {"request": request}
    )


@router.get("/dashboard/customer")
def customer_dashboard(request: Request):
    return templates.TemplateResponse(
        "dashboards/customer_dashboard.html",
        {"request": request}
    )


@router.get("/dashboard/institution")
def institution_dashboard(request: Request):
    return templates.TemplateResponse(
        "dashboards/fin_dashboard.html",
        {"request": request}
    )


@router.get("/dashboard/admin")
def admin_dashboard(request: Request):
    return templates.TemplateResponse(
        "dashboards/admin_dashboard.html",
        {"request": request}
    )


@router.get("/verify-email")
def verify_email_page(request: Request):
    return templates.TemplateResponse(
        "verify_email.html",
        {"request": request}
    )

@router.get("/forgot", response_class=HTMLResponse)
async def forgot_password_page(request: Request):
    return templates.TemplateResponse("forgot_password.html", {"request": request})