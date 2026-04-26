from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.database import Base, engine
from app.routes.auth import router as auth_router
from app.routes.pages import router as pages_router
from app.routes.stocks import router as stocks_router
from app.routes.portfolio import router as portfolio_router
from app.routes.screener import router as screener_router
from app.routes.predict import router as predict_router
from app.routes.research import router as research_router
from app.routes.news import router as news_router
from app.routes.miscellaneous import router as miscellaneous_router

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(auth_router)
app.include_router(pages_router)
app.include_router(stocks_router)
app.include_router(portfolio_router)
app.include_router(screener_router)
app.include_router(predict_router)
app.include_router(research_router)
app.include_router(news_router)
app.include_router(miscellaneous_router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

