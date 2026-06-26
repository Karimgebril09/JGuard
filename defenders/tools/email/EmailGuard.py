from __future__ import annotations

import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Optional


MAX_RECIPIENTS = 10
MAX_EMAIL_BYTES = 100_000
MAX_SENDS_PER_MINUTE = 5
MAX_SENDS_PER_DAY = 50
DUPLICATE_WINDOW_SECONDS = 120
MAX_READ_RESULTS = 25
FILTER_REQUIRED_THRESHOLD = 10

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Attachment types the agent is allowed to surface
ALLOWED_ATTACHMENT_TYPES = {".pdf", ".txt", ".csv", ".png", ".jpg", ".jpeg", ".docx", ".xlsx"}


@dataclass
class EmailDecision:
    allowed: bool
    reason: str
    audit: dict[str, Any] = field(default_factory=dict)


class EmailGuard:
    def __init__(
        self,
        *,
        allowed_mailbox: str = "me",
        allowed_domains: Optional[list[str]] = None,
        allowed_contacts: Optional[list[str]] = None,
        block_external: bool = False,
        max_recipients: int = MAX_RECIPIENTS,
        max_email_bytes: int = MAX_EMAIL_BYTES,
        max_sends_per_minute: int = MAX_SENDS_PER_MINUTE,
        max_sends_per_day: int = MAX_SENDS_PER_DAY,
        duplicate_window_seconds: int = DUPLICATE_WINDOW_SECONDS,
        max_read_results: int = MAX_READ_RESULTS,
        require_approval: bool = True,
        allowed_attachment_types: Optional[set[str]] = None,
        approval_callback: Optional[Callable[[dict[str, Any]], bool]] = None,
        audit_log_path: Optional[str] = None,
    ) -> None:
        self.allowed_mailbox = allowed_mailbox.strip().lower()
        self.allowed_domains = {d.strip().lower().lstrip("@") for d in (allowed_domains or [])}
        self.allowed_contacts = {c.strip().lower() for c in (allowed_contacts or [])}
        self.block_external = block_external
        self.max_recipients = max_recipients
        self.max_email_bytes = max_email_bytes
        self.max_sends_per_minute = max_sends_per_minute
        self.max_sends_per_day = max_sends_per_day
        self.duplicate_window_seconds = duplicate_window_seconds
        self.max_read_results = max_read_results
        self.require_approval = require_approval
        self.allowed_attachment_types = (
            {t.lower() for t in allowed_attachment_types}
            if allowed_attachment_types is not None
            else set(ALLOWED_ATTACHMENT_TYPES)
        )
        self._approval_callback = approval_callback or self._console_approval

        # State for rate limiting / dedup (thread-safe).
        self._lock = Lock()
        self._send_times: deque[float] = deque()      # rolling 1-minute window
        self._day_times: deque[float] = deque()       # rolling 24-hour window
        self._recent_sends: dict[str, float] = {}
        self.audit_records: list[dict[str, Any]] = []

        self._logger = self._build_logger(audit_log_path)

    def _build_logger(self, audit_log_path: Optional[str]) -> logging.Logger:
        logger = logging.getLogger("EmailGuard.audit")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            path = audit_log_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "email_audit.log")
            try:
                handler = logging.FileHandler(path, encoding="utf-8")
                handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
                logger.addHandler(handler)
            except OSError:
                # Fail open on logging only — never let an unwritable log block a decision.
                logger.addHandler(logging.NullHandler())
        return logger

    def _audit(
        self,
        operation: str,
        *,
        caller: str,
        recipients: list[str] | None = None,
        approval: str = "n/a",
        status: str = "",
        reason: str = "",
    ) -> dict[str, Any]:
        record = {
            "operation": operation,
            "caller": caller,
            "recipients": recipients or [],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "approval": approval,
            "status": status,
            "reason": reason,
        }
        self.audit_records.append(record)
        self._logger.info(
            "op=%s caller=%s recipients=%s approval=%s status=%s reason=%s",
            operation, caller, record["recipients"], approval, status, reason,
        )
        return record

    @staticmethod
    def _parse_addresses(raw: str | None) -> list[str]:
        if not raw:
            return []
        parts = re.split(r"[,;]", raw)
        return [p.strip() for p in parts if p.strip()]

    def _is_valid_email(self, addr: str) -> bool:
        return bool(_EMAIL_RE.match(addr))

    def _is_allowed_recipient(self, addr: str) -> bool:
        addr_l = addr.lower()
        if self.allowed_contacts and addr_l in self.allowed_contacts:
            return True
        domain = addr_l.split("@")[-1] if "@" in addr_l else ""
        if self.allowed_domains and domain in self.allowed_domains:
            return True
        if not self.allowed_contacts and not self.allowed_domains:
            return not self.block_external
        return False

    def _prune(self, q: deque[float], window: float, now: float) -> None:
        while q and now - q[0] > window:
            q.popleft()

    def validate_send_params(
        self, recipients: list[str], subject: str, body: str
    ) -> tuple[bool, str]:
        if not recipients:
            return False, "no recipients provided"
        if not subject or not subject.strip():
            return False, "subject is empty"
        if not body or not body.strip():
            return False, "body is empty"

        invalid = [r for r in recipients if not self._is_valid_email(r)]
        if invalid:
            return False, f"invalid recipient address(es): {invalid}"

        if len(recipients) > self.max_recipients:
            return False, (
                f"recipient count {len(recipients)} exceeds limit {self.max_recipients}"
            )

        size = len(subject.encode("utf-8")) + len(body.encode("utf-8"))
        if size > self.max_email_bytes:
            return False, f"email size {size} bytes exceeds limit {self.max_email_bytes}"

        return True, "parameters valid"

    def check_recipients(self, recipients: list[str]) -> tuple[bool, str]:
        blocked = [r for r in recipients if not self._is_allowed_recipient(r)]
        if blocked:
            return False, f"recipient(s) not permitted by policy: {blocked}"
        return True, "recipients permitted"

    def check_rate_limit(self) -> tuple[bool, str]:
        now = time.time()
        with self._lock:
            self._prune(self._send_times, 60.0, now)
            self._prune(self._day_times, 86400.0, now)
            minute_count = len(self._send_times)
            day_count = len(self._day_times)
            if minute_count >= self.max_sends_per_minute:
                return False, (
                    f"rate limit exceeded: {minute_count} sends in the last minute "
                    f"(max {self.max_sends_per_minute}/min)"
                )
            if day_count >= self.max_sends_per_day:
                return False, (
                    f"daily limit exceeded: {day_count} sends today "
                    f"(max {self.max_sends_per_day}/day)"
                )
        return True, "within rate limits"

    def is_duplicate(self, recipients: list[str], subject: str, body: str) -> tuple[bool, str]:
        key = self._send_key(recipients, subject, body)
        now = time.time()
        with self._lock:
            last = self._recent_sends.get(key)
            if last is not None and now - last <= self.duplicate_window_seconds:
                return True, (
                    f"duplicate send blocked: identical email sent "
                    f"{now - last:.0f}s ago (window {self.duplicate_window_seconds}s)"
                )
        return False, "not a duplicate"

    @staticmethod
    def _send_key(recipients: list[str], subject: str, body: str) -> str:
        joined = "|".join(sorted(r.lower() for r in recipients))
        return f"{joined}::{subject.strip()}::{body.strip()}"

    def request_approval(
        self, recipients: list[str], subject: str, body: str, cc: list[str], bcc: list[str]
    ) -> bool:
        if not self.require_approval:
            return True
        details = {
            "to": recipients,
            "cc": cc,
            "bcc": bcc,
            "subject": subject,
            "body": body,
        }
        return bool(self._approval_callback(details))

    @staticmethod
    def _console_approval(details: dict[str, Any]) -> bool:
        print("\n" + "=" * 60)
        print("[EmailGuard] APPROVAL REQUIRED before sending email")
        print(f"  To:      {', '.join(details['to'])}")
        if details.get("cc"):
            print(f"  Cc:      {', '.join(details['cc'])}")
        if details.get("bcc"):
            print(f"  Bcc:     {', '.join(details['bcc'])}")
        print(f"  Subject: {details['subject']}")
        print("  Body:")
        for line in str(details["body"]).splitlines() or [""]:
            print(f"    {line}")
        print("=" * 60)
        try:
            answer = input("[EmailGuard] Send this email? (yes/no): ").strip().lower()
        except (EOFError, OSError):
            return False
        return answer in {"y", "yes"}

    def guard_send(
        self,
        to: str,
        subject: str,
        body: str,
        cc: str = "",
        bcc: str = "",
        caller: str = "agent",
    ) -> EmailDecision:
        recipients = self._parse_addresses(to)
        cc_list = self._parse_addresses(cc)
        bcc_list = self._parse_addresses(bcc)
        all_recipients = recipients + cc_list + bcc_list

        def deny(reason: str, approval: str = "n/a") -> EmailDecision:
            audit = self._audit(
                "send", caller=caller, recipients=all_recipients,
                approval=approval, status="blocked", reason=reason,
            )
            print(f"[EmailGuard] SEND BLOCKED – {reason}")
            return EmailDecision(False, reason, audit)

        ok, reason = self.validate_send_params(all_recipients, subject, body)
        if not ok:
            return deny(reason)

        ok, reason = self.check_recipients(all_recipients)
        if not ok:
            return deny(reason)
        dup, reason = self.is_duplicate(all_recipients, subject, body)
        if dup:
            return deny(reason)

        ok, reason = self.check_rate_limit()
        if not ok:
            return deny(reason)

        # human approval
        approved = self.request_approval(recipients, subject, body, cc_list, bcc_list)
        if not approved:
            return deny("user declined to approve the email", approval="rejected")

        self._register_send(all_recipients, subject, body)
        audit = self._audit(
            "send", caller=caller, recipients=all_recipients,
            approval="approved", status="allowed", reason="all checks passed",
        )
        print(f"[EmailGuard] SEND APPROVED for {all_recipients}")
        return EmailDecision(True, "approved", audit)

    def _register_send(self, recipients: list[str], subject: str, body: str) -> None:
        now = time.time()
        with self._lock:
            self._send_times.append(now)
            self._day_times.append(now)
            self._prune(self._send_times, 60.0, now)
            self._prune(self._day_times, 86400.0, now)
            self._recent_sends[self._send_key(recipients, subject, body)] = now

    def record_send_result(self, decision: EmailDecision, status: str, detail: str = "") -> None:
        self._audit(
            "send_result",
            caller=str(decision.audit.get("caller", "agent")),
            recipients=list(decision.audit.get("recipients", [])),
            approval=str(decision.audit.get("approval", "n/a")),
            status=status,
            reason=detail,
        )
    def check_read(
        self,
        mailbox: str,
        max_results: int,
        query: str,
        caller: str = "agent",
    ) -> tuple[bool, str, dict[str, Any]]:
        mailbox_norm = (mailbox or "me").strip().lower()
        if mailbox_norm not in {self.allowed_mailbox, "me"}:
            reason = (
                f"mailbox '{mailbox}' is out of scope; agent is limited to "
                f"'{self.allowed_mailbox}'"
            )
            self._audit("read", caller=caller, status="blocked", reason=reason)
            print(f"[EmailGuard] READ BLOCKED – {reason}")
            return False, reason, {}

        # query restrictions.
        requested = max_results if isinstance(max_results, int) and max_results > 0 else 10
        clamped = min(requested, self.max_read_results)
        has_filter = bool(query and query.strip())
        if clamped > FILTER_REQUIRED_THRESHOLD and not has_filter:
            reason = (
                f"reading {clamped} emails requires a filter "
                f"(date range, sender, folder, etc.)"
            )
            self._audit("read", caller=caller, status="blocked", reason=reason)
            print(f"[EmailGuard] READ BLOCKED – {reason}")
            return False, reason, {}

        self._audit(
            "read", caller=caller, status="allowed",
            reason=f"max_results={clamped} filter={'yes' if has_filter else 'no'}",
        )
        return True, "read permitted", {"max_results": clamped, "query": query or "", "mailbox": self.allowed_mailbox}

    def filter_attachments(self, attachments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        safe: list[dict[str, Any]] = []
        for att in attachments or []:
            name = str(att.get("filename", att.get("name", "")))
            ext = os.path.splitext(name)[1].lower()
            entry = dict(att)
            entry["opened"] = False  # never auto-open
            if ext not in self.allowed_attachment_types:
                entry["blocked"] = True
                entry["reason"] = f"attachment type '{ext or 'unknown'}' is not allowed"
                print(f"[EmailGuard] Attachment blocked: {name} ({ext or 'unknown'})")
            else:
                entry["blocked"] = False
                entry["scan_required"] = True
            safe.append(entry)
        return safe

    def sanitize_emails(self, emails: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sanitized: list[dict[str, Any]] = []
        for email in emails:
            if "error" in email:
                sanitized.append(email)
                continue
            safe = dict(email)

            if email.get("attachments"):
                safe["attachments"] = self.filter_attachments(email["attachments"])

            sanitized.append(safe)
        return sanitized


# if __name__ == "__main__":
#     guard = EmailGuard(require_approval=False, allowed_domains=["example.com"])

#     print("\n--- valid send ---")
#     print(guard.guard_send("bob@example.com", "Hello", "Hi there, this is a test."))

#     print("\n--- blocked: empty subject ---")
#     print(guard.guard_send("bob@example.com", "", "Body only"))

#     print("\n--- blocked: off-policy domain ---")
#     print(guard.guard_send("attacker@evil.com", "Hi", "Body"))

#     print("\n--- duplicate ---")
#     print(guard.guard_send("carol@example.com", "Report", "Numbers attached"))
#     print(guard.guard_send("carol@example.com", "Report", "Numbers attached"))

#     print("\n--- read sanitization (attachments) ---")
#     emails = [{
#         "id": "x1",
#         "from": "newsletter@spam.com",
#         "subject": "URGENT",
#         "body": "Quarterly report attached.",
#         "snippet": "Quarterly report...",
#         "attachments": [{"filename": "report.pdf"}, {"filename": "run.exe"}],
#     }]
#     for e in guard.sanitize_emails(emails):
#         print(e.get("attachments"))
