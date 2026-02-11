"""
KEEP – Email Service
Async email delivery using SMTP with SSL for the Health Vault platform.
Uses smtplib (stdlib) for zero-dependency simplicity.
"""

import asyncio
import logging
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import settings
from email_templates import get_welcome_email_html

logger = logging.getLogger(__name__)


class EmailService:
    """
    Async-friendly email service backed by smtplib + SSL.

    Usage:
        await email_service.send_welcome_email("user@gmail.com", "Alice")
    """

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_message(
        self,
        to_email: str,
        subject: str,
        html_body: str,
    ) -> MIMEMultipart:
        """Build a MIME email message."""
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{settings.SMTP_FROM_NAME} <{settings.SMTP_FROM_EMAIL}>"
        msg["To"] = to_email
        msg["Subject"] = subject
        msg["X-Mailer"] = "KEEP Health Vault"

        # Plain-text fallback
        plain_text = (
            f"Welcome to KEEP!\n\n"
            f"Your personal health vault is ready.\n"
            f"Visit https://app.onkeep.co to get started.\n\n"
            f"— The KEEP Team"
        )
        msg.attach(MIMEText(plain_text, "plain", "utf-8"))
        msg.attach(MIMEText(html_body, "html", "utf-8"))
        return msg

    def _send_smtp(self, msg: MIMEMultipart, to_email: str) -> None:
        """Blocking SMTP send over SSL (port 465)."""
        context = ssl.create_default_context()

        with smtplib.SMTP_SSL(
            settings.SMTP_HOST,
            settings.SMTP_PORT,
            context=context,
            timeout=15,
        ) as server:
            server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_FROM_EMAIL, to_email, msg.as_string())

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    async def send_welcome_email(self, to_email: str, first_name: str) -> bool:
        """
        Send the onboarding welcome email.

        Runs the blocking SMTP call in a thread executor so it never blocks
        the FastAPI event loop.

        Args:
            to_email:    Recipient email address.
            first_name:  User's first name for personalization.

        Returns:
            True if the email was sent successfully, False otherwise.
        """
        # Guard: skip sending if SMTP is not configured
        if not settings.SMTP_USERNAME or not settings.SMTP_PASSWORD:
            logger.warning("SMTP not configured — skipping welcome email for %s", to_email)
            return False

        try:
            subject = "Welcome to KEEP | Your health journey starts here 🛡️"
            html_body = get_welcome_email_html(first_name)
            msg = self._build_message(to_email, subject, html_body)

            # Offload blocking I/O to the default thread executor
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._send_smtp, msg, to_email)

            logger.info("✅ Welcome email sent to %s", to_email)
            return True

        except smtplib.SMTPAuthenticationError:
            logger.error("❌ SMTP auth failed — check SMTP_USERNAME / SMTP_PASSWORD")
        except smtplib.SMTPConnectError:
            logger.error("❌ Could not connect to SMTP server %s:%s", settings.SMTP_HOST, settings.SMTP_PORT)
        except Exception as exc:
            logger.error("❌ Failed to send welcome email to %s: %s", to_email, exc, exc_info=True)

        return False


# Singleton instance – import this everywhere
email_service = EmailService()
