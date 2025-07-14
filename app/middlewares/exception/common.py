from enum import Enum
from typing import Optional


class ErrorCode(int, Enum):
    SystemError = 1000,
    RuntimeError = 1001,
    ValidationError = 1002,
    ValueError = 1003,
    UnknownError = 1004,
    ObjectNotExist = 1005,
    BadAuthentication = 1006,
    UnauthorizedAppid = 1007,
    ParameterRequired = 1008,
    ParameterInvalid = 1009,
    AccessDenied = 1010,
    ServiceError = 1011,
    CosError = 1012,
    CompanyIdRequired = 1013


class ErrorMessage(str, Enum):
    SystemError = "System error",
    RuntimeError = "Runtime error",
    ValueError = "Value error",
    ValidationError = "Validation error",
    UnknownError = "Unknown error",
    ObjectNotExist = "Object not exist",
    BadAuthentication = 'Bad authentication',
    UnauthorizedAppid = 'Unauthorized appid',
    ParameterRequired = 'Parameter required',
    ParameterInvalid = 'Parameter invalid',
    AccessDenied = 'Access denied',
    ServiceError = 'Service error',
    CosError = 'Cos error',
    CompanyIdRequired = 'Company id required',


class CommonException(Exception):
    def __init__(self, errcode: ErrorCode, errmsg: ErrorMessage, detail: str = None):
        self.errcode = errcode
        self.errmsg = errmsg
        self.detail = detail

    def __str__(self):
        return f'errcode: {self.errcode.value}, errmsg: {self.errmsg.value}, detail: {self.detail}'

    @classmethod
    def object_not_exist(cls, detail: Optional[str] = None):
        return cls(errcode=ErrorCode.ObjectNotExist, errmsg=ErrorMessage.ObjectNotExist, detail=detail)

    @classmethod
    def system_error(cls, detail: Optional[str] = None):
        return cls(errcode=ErrorCode.SystemError, errmsg=ErrorMessage.SystemError, detail=detail)

    @classmethod
    def unauthorized_appid(cls, detail: Optional[str] = None):
        return cls(errcode=ErrorCode.UnauthorizedAppid, errmsg=ErrorMessage.UnauthorizedAppid, detail=detail)

    @classmethod
    def parameter_required(cls, detail: Optional[str] = None):
        return cls(errcode=ErrorCode.ParameterRequired, errmsg=ErrorMessage.ParameterRequired, detail=detail)

    @classmethod
    def parameter_invalid(cls, detail: Optional[str] = None):
        return cls(errcode=ErrorCode.ParameterInvalid, errmsg=ErrorMessage.ParameterInvalid, detail=detail)

    @classmethod
    def runtime_error(cls, detail: Optional[str] = None):
        return cls(errcode=ErrorCode.RuntimeError, errmsg=ErrorMessage.RuntimeError, detail=detail)

    @classmethod
    def access_denied(cls, detail: Optional[str] = None):
        return cls(errcode=ErrorCode.AccessDenied, errmsg=ErrorMessage.AccessDenied, detail=detail)

    @classmethod
    def service_error(cls, detail: Optional[str] = None):
        return cls(errcode=ErrorCode.ServiceError, errmsg=ErrorMessage.ServiceError, detail=detail)

    @classmethod
    def cos_error(cls, detail: Optional[str] = None):
        return cls(errcode=ErrorCode.CosError, errmsg=ErrorMessage.CosError, detail=detail)

    @classmethod
    def company_id_required(cls, detail: Optional[str] = None):
        return cls(errcode=ErrorCode.CompanyIdRequired, errmsg=ErrorMessage.CompanyIdRequired, detail=detail)
