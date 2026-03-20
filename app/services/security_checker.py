"""安全检查服务 - 检测代码中的敏感信息

参考 Repomix 的 Secretlint 实现，使用正则表达式检测常见的敏感信息类型。
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SecurityCheckResult:
    """安全检查结果"""
    file_path: str
    line_number: int
    message: str
    severity: str  # "error", "warning", "info"
    rule_id: str


class SecurityChecker:
    """安全检查器 - 检测代码中的敏感信息"""
    
    # 敏感信息检测规则
    RULES = [
        # AWS
        {
            "id": "aws-access-key",
            "pattern": r"(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}",
            "message": "检测到 AWS Access Key ID",
            "severity": "error",
        },
        {
            "id": "aws-secret-key",
            "pattern": r"(?:aws)?_?secret_?(?:access)?_?key['\"]?\s*[:=]\s*['\"][A-Za-z0-9/+=]{40}['\"]",
            "message": "检测到 AWS Secret Access Key",
            "severity": "error",
        },
        # GitHub
        {
            "id": "github-token",
            "pattern": r"github[_\-]?token['\"]?\s*[:=]\s*['\"](ghp_[A-Za-z0-9_]{36}|gho_[A-Za-z0-9_]{36}|ghu_[A-Za-z0-9_]{36}|ghs_[A-Za-z0-9_]{36}|ghr_[A-Za-z0-9_]{36})['\"]",
            "message": "检测到 GitHub Token",
            "severity": "error",
        },
        {
            "id": "github-oauth",
            "pattern": r"github[_\-]?oauth[_\-]?token['\"]?\s*[:=]\s*['\"][A-Za-z0-9_]{35,40}['\"]",
            "message": "检测到 GitHub OAuth Token",
            "severity": "error",
        },
        # Slack
        {
            "id": "slack-token",
            "pattern": r"xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}",
            "message": "检测到 Slack Token",
            "severity": "error",
        },
        {
            "id": "slack-webhook",
            "pattern": r"https://hooks\.slack\.com/services/T[A-Z0-9]{8,10}/B[A-Z0-9]{8,10}/[A-Za-z0-9]{24}",
            "message": "检测到 Slack Webhook URL",
            "severity": "error",
        },
        # Google
        {
            "id": "google-api-key",
            "pattern": r"AIza[A-Za-z0-9_\-]{35}",
            "message": "检测到 Google API Key",
            "severity": "error",
        },
        {
            "id": "google-oauth",
            "pattern": r"[0-9]+-[A-Za-z0-9_]{32}\.apps\.googleusercontent\.com",
            "message": "检测到 Google OAuth Client ID",
            "severity": "warning",
        },
        # Stripe
        {
            "id": "stripe-api-key",
            "pattern": r"sk_live_[0-9a-zA-Z]{24}",
            "message": "检测到 Stripe Live API Key",
            "severity": "error",
        },
        {
            "id": "stripe-publishable-key",
            "pattern": r"pk_live_[0-9a-zA-Z]{24}",
            "message": "检测到 Stripe Publishable Key",
            "severity": "warning",
        },
        # Twilio
        {
            "id": "twilio-account-sid",
            "pattern": r"AC[a-f0-9]{32}",
            "message": "检测到 Twilio Account SID",
            "severity": "warning",
        },
        {
            "id": "twilio-auth-token",
            "pattern": r"twilio[_\-]?auth[_\-]?token['\"]?\s*[:=]\s*['\"][a-f0-9]{32}['\"]",
            "message": "检测到 Twilio Auth Token",
            "severity": "error",
        },
        # 数据库连接串
        {
            "id": "database-url",
            "pattern": r"(?:mysql|postgres|mongodb|redis)://[^\s'\"]+:[^\s'\"]+@[^\s'\"]+",
            "message": "检测到数据库连接字符串（包含密码）",
            "severity": "error",
        },
        # 私钥
        {
            "id": "private-key",
            "pattern": r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----",
            "message": "检测到私钥",
            "severity": "error",
        },
        # SSH
        {
            "id": "ssh-private-key",
            "pattern": r"-----BEGIN OPENSSH PRIVATE KEY-----",
            "message": "检测到 SSH 私钥",
            "severity": "error",
        },
        # 通用密码/API Key
        {
            "id": "generic-password",
            "pattern": r"(?:password|passwd|pwd)['\"]?\s*[:=]\s*['\"][^'\"]{8,}['\"]",
            "message": "检测到可能的密码",
            "severity": "warning",
        },
        {
            "id": "generic-api-key",
            "pattern": r"(?:api[_\-]?key|apikey|api[_\-]?secret)['\"]?\s*[:=]\s*['\"][A-Za-z0-9_\-]{20,}['\"]",
            "message": "检测到可能的 API Key",
            "severity": "warning",
        },
        {
            "id": "generic-secret",
            "pattern": r"(?:secret|secret[_\-]?key|access[_\-]?token|auth[_\-]?token)['\"]?\s*[:=]\s*['\"][A-Za-z0-9_\-]{20,}['\"]",
            "message": "检测到可能的密钥/Token",
            "severity": "warning",
        },
        # JWT
        {
            "id": "jwt-token",
            "pattern": r"eyJ[A-Za-z0-9_\-]*\.eyJ[A-Za-z0-9_\-]*\.[A-Za-z0-9_\-]*",
            "message": "检测到 JWT Token",
            "severity": "warning",
        },
        # Bearer Token
        {
            "id": "bearer-token",
            "pattern": r"Bearer\s+[A-Za-z0-9_\-\.]{20,}",
            "message": "检测到 Bearer Token",
            "severity": "warning",
        },
    ]
    
    # 编译正则表达式
    _compiled_rules: list[dict] = []
    
    @classmethod
    def _get_compiled_rules(cls) -> list[dict]:
        """获取编译后的规则"""
        if not cls._compiled_rules:
            for rule in cls.RULES:
                compiled = rule.copy()
                compiled["regex"] = re.compile(rule["pattern"], re.IGNORECASE)
                cls._compiled_rules.append(compiled)
        return cls._compiled_rules
    
    @classmethod
    def check_content(cls, content: str, file_path: str) -> list[SecurityCheckResult]:
        """检查内容中的敏感信息
        
        Args:
            content: 文件内容
            file_path: 文件路径
            
        Returns:
            检测到的敏感信息列表
        """
        results = []
        rules = cls._get_compiled_rules()
        
        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            for rule in rules:
                matches = rule["regex"].finditer(line)
                for match in matches:
                    results.append(SecurityCheckResult(
                        file_path=file_path,
                        line_number=line_num,
                        message=rule["message"],
                        severity=rule["severity"],
                        rule_id=rule["id"],
                    ))
        
        return results
    
    @classmethod
    def check_file(cls, file_path: str) -> list[SecurityCheckResult]:
        """检查文件中的敏感信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            检测到的敏感信息列表
        """
        try:
            with open(file_path, "r", errors="ignore") as f:
                content = f.read()
            return cls.check_content(content, file_path)
        except Exception as e:
            logger.debug("Failed to check file %s: %s", file_path, e)
            return []
    
    @classmethod
    def check_files(cls, file_paths: list[str]) -> dict[str, list[SecurityCheckResult]]:
        """批量检查多个文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            文件路径 -> 检测结果的映射
        """
        results = {}
        for file_path in file_paths:
            file_results = cls.check_file(file_path)
            if file_results:
                results[file_path] = file_results
        return results
    
    @classmethod
    def has_sensitive_info(cls, content: str) -> bool:
        """快速检查内容是否包含敏感信息
        
        Args:
            content: 文件内容
            
        Returns:
            是否包含敏感信息
        """
        rules = cls._get_compiled_rules()
        for rule in rules:
            if rule["regex"].search(content):
                return True
        return False
    
    @classmethod
    def redact_sensitive(cls, content: str, replacement: str = "***REDACTED***") -> str:
        """脱敏处理内容中的敏感信息
        
        Args:
            content: 原始内容
            replacement: 替换字符串
            
        Returns:
            脱敏后的内容
        """
        rules = cls._get_compiled_rules()
        redacted = content
        for rule in rules:
            redacted = rule["regex"].sub(replacement, redacted)
        return redacted


# 全局实例
security_checker = SecurityChecker()


def check_file_security(file_path: str) -> list[SecurityCheckResult]:
    """检查文件安全性的便捷函数"""
    return security_checker.check_file(file_path)


def check_content_security(content: str, file_path: str = "unknown") -> list[SecurityCheckResult]:
    """检查内容安全性的便捷函数"""
    return security_checker.check_content(content, file_path)


def has_sensitive_info(content: str) -> bool:
    """快速检查是否包含敏感信息的便捷函数"""
    return security_checker.has_sensitive_info(content)


def redact_sensitive(content: str) -> str:
    """脱敏处理的便捷函数"""
    return security_checker.redact_sensitive(content)