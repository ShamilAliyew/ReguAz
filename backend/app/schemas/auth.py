"""Strict request and response contracts for cookie-based authentication."""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        value = value.strip().casefold()
        if not _EMAIL_PATTERN.fullmatch(value):
            raise ValueError("Etibarlı e-poçt ünvanı daxil edin.")
        return value


class RegisterRequest(LoginRequest):
    name: str = Field(min_length=2, max_length=120)
    password: str = Field(min_length=10, max_length=128)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 2:
            raise ValueError("Ad ən azı 2 simvol olmalıdır.")
        return value

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        if value.isspace():
            raise ValueError("Şifrə yalnız boşluqlardan ibarət ola bilməz.")
        return value


class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AuthResponse(BaseModel):
    user: UserResponse
