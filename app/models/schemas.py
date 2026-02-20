from pydantic import BaseModel, ConfigDict
from datetime import datetime
from app.models.enums import UserRole, InstanceState


# --- Auth ---

class RegisterRequest(BaseModel):
    username: str
    password: str
    registration_code: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str


# --- User ---

class UserResponse(BaseModel):
    id: str
    username: str
    role: UserRole
    created_at: datetime
    max_invite_codes: int = 2

    model_config = ConfigDict(from_attributes=True)


class UpdateUserRoleRequest(BaseModel):
    role: UserRole


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


# --- Registration Codes ---

class GenerateCodeRequest(BaseModel):
    expires_in_hours: int = 24


class RegistrationCodeResponse(BaseModel):
    code: str
    created_at: datetime
    expires_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RegistrationCodeDetailResponse(BaseModel):
    id: str
    code: str
    created_at: datetime
    expires_at: datetime
    used_at: datetime | None
    used_by: str | None
    created_by_username: str | None = None
    is_valid: bool

    model_config = ConfigDict(from_attributes=True)


class UpdateInviteLimitRequest(BaseModel):
    max_invite_codes: int


# --- Miner Instance ---

class InstanceCreate(BaseModel):
    pass  # ID wird serverseitig erzeugt, keine Pflichtfelder


class InstanceResponse(BaseModel):
    id: str
    user_id: str
    status: InstanceState
    container_id: str | None = None
    port: int | None = None
    ui_url: str | None = None
    created_at: datetime
    last_started_at: datetime | None = None
    last_stopped_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class InstanceStatus(BaseModel):
    id: str
    status: InstanceState
    container_id: str | None = None
    port: int | None = None
