from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import ALLOWED_ORIGINS

app = FastAPI(title="TruthLens AI", description="A probabilistic news-text classifier, not a fact-checker.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(router)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_: Request, exc: StarletteHTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        payload = detail
    else:
        code = {400: "BAD_REQUEST", 404: "NOT_FOUND", 422: "VALIDATION_ERROR", 503: "SERVICE_UNAVAILABLE"}.get(exc.status_code, "HTTP_ERROR")
        payload = {"error": str(detail), "code": code}
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, __: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "Please provide a non-empty headline, article, and valid model.", "code": "VALIDATION_ERROR"})
