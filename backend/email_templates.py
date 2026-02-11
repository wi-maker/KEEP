"""
KEEP – Email Templates
Premium HTML email templates for the Health Vault platform.
All templates use inline CSS for maximum email client compatibility.
"""


def get_welcome_email_html(first_name: str) -> str:
    """
    Generate the Welcome onboarding email HTML.

    Args:
        first_name: The user's first name for personalization.

    Returns:
        Fully rendered HTML string with inline styles.
    """
    return f"""<!DOCTYPE html>
<html lang="en" xmlns="http://www.w3.org/1999/xhtml">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="X-UA-Compatible" content="IE=edge" />
  <title>Welcome to KEEP</title>
  <!--[if mso]>
  <noscript>
    <xml>
      <o:OfficeDocumentSettings>
        <o:PixelsPerInch>96</o:PixelsPerInch>
      </o:OfficeDocumentSettings>
    </xml>
  </noscript>
  <![endif]-->
</head>
<body style="margin:0;padding:0;background-color:#f1f5f9;font-family:'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;-webkit-font-smoothing:antialiased;">

  <!-- Outer wrapper -->
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f1f5f9;">
    <tr>
      <td align="center" style="padding:40px 16px;">

        <!-- Email card -->
        <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;background-color:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.06);">

          <!-- ============ HEADER ============ -->
          <tr>
            <td style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#0f172a 100%);padding:40px 40px 32px 40px;text-align:center;">
              <!-- Logo -->
              <img
                src="https://onkeep.co/logo.png"
                alt="KEEP"
                width="120"
                style="display:inline-block;width:120px;height:auto;"
              />
              <p style="margin:16px 0 0 0;font-size:13px;letter-spacing:0.08em;text-transform:uppercase;color:#94a3b8;font-weight:600;">
                Your Personal Health Vault
              </p>
            </td>
          </tr>

          <!-- ============ GREETING ============ -->
          <tr>
            <td style="padding:36px 40px 8px 40px;">
              <h1 style="margin:0;font-size:24px;font-weight:700;color:#0f172a;letter-spacing:-0.02em;">
                Hi {first_name},
              </h1>
            </td>
          </tr>

          <!-- ============ BODY COPY ============ -->
          <tr>
            <td style="padding:8px 40px 24px 40px;">
              <p style="margin:0 0 16px 0;font-size:15px;line-height:1.7;color:#475569;">
                Welcome to <strong style="color:#0f172a;">KEEP</strong>&mdash;the last place you'll ever need to store your medical history. We've built your personal health vault to ensure that your records are as mobile and accessible as you are.
              </p>
            </td>
          </tr>

          <!-- ============ FEATURE CARDS ============ -->
          <tr>
            <td style="padding:0 40px;">
              <p style="margin:0 0 16px 0;font-size:15px;font-weight:600;color:#0f172a;">
                What can you do with KEEP?
              </p>
            </td>
          </tr>

          <!-- Feature 1: Centralize -->
          <tr>
            <td style="padding:0 40px 12px 40px;">
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f8fafc;border-radius:12px;border:1px solid #e2e8f0;">
                <tr>
                  <td style="padding:20px 24px;">
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                      <tr>
                        <td width="44" valign="top">
                          <div style="width:40px;height:40px;background:linear-gradient(135deg,#0d9488,#14b8a6);border-radius:10px;text-align:center;line-height:40px;font-size:18px;">
                            &#128193;
                          </div>
                        </td>
                        <td style="padding-left:16px;">
                          <p style="margin:0 0 4px 0;font-size:14px;font-weight:700;color:#0f172a;">Centralize Your Data</p>
                          <p style="margin:0;font-size:13px;line-height:1.6;color:#64748b;">Upload lab tests, prescriptions, and scans. No more lost paperwork.</p>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Feature 2: AI Analysis -->
          <tr>
            <td style="padding:0 40px 12px 40px;">
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f8fafc;border-radius:12px;border:1px solid #e2e8f0;">
                <tr>
                  <td style="padding:20px 24px;">
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                      <tr>
                        <td width="44" valign="top">
                          <div style="width:40px;height:40px;background:linear-gradient(135deg,#7c3aed,#a78bfa);border-radius:10px;text-align:center;line-height:40px;font-size:18px;">
                            &#129504;
                          </div>
                        </td>
                        <td style="padding-left:16px;">
                          <p style="margin:0 0 4px 0;font-size:14px;font-weight:700;color:#0f172a;">Understand Your Health</p>
                          <p style="margin:0;font-size:13px;line-height:1.6;color:#64748b;">Our AI analyzes your records to explain complex medical jargon in plain English.</p>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- Feature 3: Secure Sharing -->
          <tr>
            <td style="padding:0 40px 28px 40px;">
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f8fafc;border-radius:12px;border:1px solid #e2e8f0;">
                <tr>
                  <td style="padding:20px 24px;">
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                      <tr>
                        <td width="44" valign="top">
                          <div style="width:40px;height:40px;background:linear-gradient(135deg,#0369a1,#38bdf8);border-radius:10px;text-align:center;line-height:40px;font-size:18px;">
                            &#128274;
                          </div>
                        </td>
                        <td style="padding-left:16px;">
                          <p style="margin:0 0 4px 0;font-size:14px;font-weight:700;color:#0f172a;">Share with Confidence</p>
                          <p style="margin:0;font-size:13px;line-height:1.6;color:#64748b;">Generate secure, time-limited links to show your doctors or family your history instantly.</p>
                        </td>
                      </tr>
                    </table>
                  </td>
                </tr>
              </table>
            </td>
          </tr>

          <!-- ============ CTA SECTION ============ -->
          <tr>
            <td style="padding:0 40px 12px 40px;">
              <p style="margin:0;font-size:14px;line-height:1.6;color:#475569;">
                <strong style="color:#0f172a;">Take the first step:</strong> Your vault is empty! Upload your first document today to see the AI analysis in action.
              </p>
            </td>
          </tr>

          <tr>
            <td style="padding:12px 40px 36px 40px;text-align:center;">
              <a href="https://app.onkeep.co/"
                 target="_blank"
                 style="display:inline-block;background:linear-gradient(135deg,#0d9488,#0f766e);color:#ffffff;text-decoration:none;padding:14px 36px;border-radius:10px;font-size:15px;font-weight:600;letter-spacing:0.02em;box-shadow:0 4px 14px rgba(13,148,136,0.35);">
                Upload My First Record &rarr;
              </a>
            </td>
          </tr>

          <!-- ============ DIVIDER ============ -->
          <tr>
            <td style="padding:0 40px;">
              <div style="height:1px;background-color:#e2e8f0;"></div>
            </td>
          </tr>

          <!-- ============ FOOTER ============ -->
          <tr>
            <td style="padding:28px 40px 36px 40px;text-align:center;">
              <p style="margin:0 0 6px 0;font-size:14px;color:#475569;">
                Stay healthy,
              </p>
              <p style="margin:0 0 20px 0;font-size:14px;font-weight:700;color:#0f172a;">
                The KEEP Team
              </p>
              <p style="margin:0;font-size:11px;color:#94a3b8;line-height:1.6;">
                &copy; {_get_current_year()} KEEP Health&ensp;|&ensp;
                <a href="https://onkeep.co" style="color:#94a3b8;text-decoration:underline;">onkeep.co</a>
              </p>
              <p style="margin:8px 0 0 0;font-size:11px;color:#cbd5e1;">
                You received this email because you signed up for KEEP.
              </p>
            </td>
          </tr>

        </table>
        <!-- /Email card -->

      </td>
    </tr>
  </table>

</body>
</html>"""


def _get_current_year() -> int:
    """Return the current year for the copyright footer."""
    from datetime import datetime
    return datetime.now().year
