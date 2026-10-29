import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Literal

import redis
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from sentinel.config import Config, get_config
from sentinel.queue import enqueue
from sentinel.store import Principal, Store

SCOPES = {"messages:write", "moderation:read", "review:write"}

_bearer = HTTPBearer(auto_error=False)


class MessageIn(BaseModel):
    message_id: str = Field(min_length=1, max_length=128)
    conversation_id: str = Field(min_length=1, max_length=128)
    author_id: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1)
    created_at: datetime


class ReviewIn(BaseModel):
    decision: Literal["confirm", "reject"]
    # Defaults: confirm keeps the model verdict, reject means the message was fine ("ok").
    final_verdict: Literal["ok", "flag", "hide"] | None = None
    category: str | None = Field(None, max_length=64)
    reviewer_id: str | None = Field(None, max_length=128)
    notes: str | None = Field(None, max_length=2000)


def create_app(config: Config | None = None, store: Store | None = None, queue: redis.Redis | None = None) -> FastAPI:
    config = config or get_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store or Store(config.database_path)
        app.state.redis = queue or redis.Redis.from_url(config.redis_url, decode_responses=True)
        yield

    # Handlers and dependencies are `async def` on purpose: they run on the event loop thread only, so
    # the single SQLite connection is never used concurrently. Every call is a sub-millisecond query.
    app = FastAPI(title="chat-sentinel", version="0.1.0", lifespan=lifespan)

    def require(scope: str) -> Callable[..., Principal]:
        async def dependency(
            request: Request, credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)]
        ) -> Principal:
            unauthorized = HTTPException(
                status.HTTP_401_UNAUTHORIZED, "invalid or missing API key", {"WWW-Authenticate": "Bearer"}
            )
            if credentials is None:
                raise unauthorized
            principal = request.app.state.store.principal(credentials.credentials)
            if principal is None:
                raise unauthorized
            if scope not in principal.scopes:
                raise HTTPException(status.HTTP_403_FORBIDDEN, f"API key lacks scope {scope}")
            _rate_limit(request.app.state.redis, principal, config.api.rate_limit_per_minute)
            return principal

        return dependency

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/messages", status_code=status.HTTP_202_ACCEPTED)
    async def post_message(
        body: MessageIn, request: Request, principal: Annotated[Principal, Depends(require("messages:write"))]
    ) -> dict:
        if len(body.text) > config.api.max_text_length:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "text too long")
        key, created = request.app.state.store.insert_message(
            principal.client_id, body.message_id, body.conversation_id, body.author_id, body.text, body.created_at
        )
        if created:
            enqueue(request.app.state.redis, key)
        return {"message_id": body.message_id, "status": "queued" if created else "duplicate"}

    @app.get("/moderation/{message_id}")
    async def get_moderation(
        message_id: str, request: Request, principal: Annotated[Principal, Depends(require("moderation:read"))]
    ) -> dict:
        result = request.app.state.store.moderation(principal.client_id, message_id)
        if result is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "message not found")
        return result

    @app.get("/review-queue")
    async def get_review_queue(
        request: Request,
        principal: Annotated[Principal, Depends(require("moderation:read"))],
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> dict:
        return {"items": request.app.state.store.review_queue(principal.client_id, limit)}

    @app.post("/review/{message_id}")
    async def post_review(
        message_id: str,
        body: ReviewIn,
        request: Request,
        principal: Annotated[Principal, Depends(require("review:write"))],
    ) -> dict:
        review = request.app.state.store.save_review(principal.client_id, message_id, **body.model_dump())
        if review is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "message not found or not moderated yet")
        return review

    return app


def _rate_limit(client: redis.Redis, principal: Principal, per_minute: int) -> None:
    window = int(time.time() // 60)
    key = f"sentinel:rl:{principal.key_id}:{window}"
    count = client.incr(key)
    if count == 1:
        client.expire(key, 61)
    if count > per_minute:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded", {"Retry-After": "60"})
