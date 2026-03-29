from datetime import datetime
from typing import Optional, Union

from pydantic import BaseModel, field_validator


EXPIRES_IN_MAP = {
    "1h": 3600,
    "24h": 86400,
    "7d": 604800,
}


class SecretCreate(BaseModel):
    payload: str
    expires_in: str = "24h"
    max_views: Union[int, str] = 1

    @field_validator("expires_in")
    @classmethod
    def validate_expires_in(cls, v):
        if v not in EXPIRES_IN_MAP:
            raise ValueError(f"expires_in must be one of: {list(EXPIRES_IN_MAP.keys())}")
        return v

    @field_validator("max_views")
    @classmethod
    def validate_max_views(cls, v):
        if v == "unlimited":
            return None
        if isinstance(v, str):
            try:
                v = int(v)
            except ValueError:
                raise ValueError("max_views must be a positive integer or 'unlimited'")
        if isinstance(v, int) and v < 1:
            raise ValueError("max_views must be >= 1 or 'unlimited'")
        return v


class SecretCreateResponse(BaseModel):
    link: str
    creator_token: str
    expires_at: str


class SecretViewResponse(BaseModel):
    payload: str
    content_type: str = "text/plain"


class RevokeRequest(BaseModel):
    creator_token: str


class HealthResponse(BaseModel):
    status: str
    db: str
