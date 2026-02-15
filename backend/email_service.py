"""
KEEP – Email Service
Async email delivery via Resend HTTP API for the Health Vault platform.
Uses httpx (already in requirements) — no SMTP, no blocked ports.
"""

import logging
import httpx

from config import settings
from email_templates import get_welcome_email_html

logger = logging.getLogger(__name__)

# Resend API endpoint
RESEND_API_URL = "https://api.resend.com/emails"


class EmailService:
    """
    Email service backed by Resend HTTP API.

    Why not SMTP?
    Railway (and most cloud platforms) block outbound ports 25/465/587.
    Resend uses HTTPS (port 443) which is never blocked.

    Usage:
        await email_service.send_welcome_email("user@gmail.com", "Alice")
    """

    async def send_welcome_email(self, to_email: str, first_name: str) -> bool:
        """
        Send the onboarding welcome email via Resend.

        Args:
            to_email:    Recipient email address.
            first_name:  User's first name for personalization.

        Returns:
            True if the email was sent successfully, False otherwise.
        """
        # Guard: skip if API key not configured
        if not settings.RESEND_API_KEY:
            logger.warning("RESEND_API_KEY not configured — skipping welcome email for %s", to_email)
            return False

        try:
            subject = "Welcome to KEEP"
            html_body = get_welcome_email_html(first_name)

            payload = {
                "from": f"{settings.EMAIL_FROM_NAME} <{settings.EMAIL_FROM_ADDRESS}>",
                "to": [to_email],
                "subject": subject,
                "html": html_body,
            }

            # Set reply-to if configured (prevents 550 bounces on domains without MX records)
            if settings.REPLY_TO_EMAIL:
                payload["reply_to"] = settings.REPLY_TO_EMAIL

            headers = {
                "Authorization": f"Bearer {settings.RESEND_API_KEY}",
                "Content-Type": "application/json",
            }


            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(RESEND_API_URL, json=payload, headers=headers)

            if response.status_code in (200, 201):
                logger.info("✅ Welcome email sent to %s (Resend ID: %s)", to_email, response.json().get("id"))
                return True
            else:
                logger.error(
                    "❌ Resend API error %s for %s: %s",
                    response.status_code, to_email, response.text
                )
                return False

        except httpx.TimeoutException:
            logger.error("❌ Resend API timeout sending to %s", to_email)
        except Exception as exc:
            logger.error("❌ Failed to send welcome email to %s: %s", to_email, exc, exc_info=True)

        return False


# Singleton instance
email_service = EmailService()
