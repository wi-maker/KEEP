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
<body style="margin:0;padding:0;background-color:#f0fdf4;font-family:'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;-webkit-font-smoothing:antialiased;">

  <!-- Outer wrapper -->
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f0fdf4;">
    <tr>
      <td align="center" style="padding:40px 16px;">

        <!-- Email card -->
        <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;background-color:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.06);">

          <!-- ============ HEADER ============ -->
          <tr>
            <td style="background:linear-gradient(135deg,#0d9488 0%,#0f766e 50%,#115e59 100%);padding:40px 40px 32px 40px;text-align:center;">
              <!-- Logo -->
              <img
                src="https://onkeep.co/logo.png"
                alt="KEEP"
                width="120"
                style="display:inline-block;width:120px;height:auto;"
              />
              <p style="margin:16px 0 0 0;font-size:13px;letter-spacing:0.08em;text-transform:uppercase;color:rgba(255,255,255,0.7);font-weight:600;">
                Your Personal Health Vault
              </p>
            </td>
          </tr>

          <!-- ============ GREETING ============ -->
          <tr>
            <td style="padding:36px 40px 8px 40px;">
              <p style="margin:0;font-size:16px;line-height:1.7;color:#1e293b;">
                Hi {first_name},
              </p>
            </td>
          </tr>

          <!-- ============ HEADLINE ============ -->
          <tr>
            <td style="padding:8px 40px 8px 40px;">
              <h1 style="margin:0;font-size:24px;font-weight:700;color:#0f766e;letter-spacing:-0.02em;">
                Welcome to KEEP &ndash; your personal health vault.
              </h1>
            </td>
          </tr>

          <!-- ============ INTRO ============ -->
          <tr>
            <td style="padding:12px 40px 28px 40px;">
              <p style="margin:0;font-size:15px;line-height:1.7;color:#475569;">
                KEEP stores all your medical records in one secure place, so you no longer have to deal with lost results or scattered paperwork.
              </p>
            </td>
          </tr>

          <!-- ============ HOW IT WORKS HEADING ============ -->
          <tr>
            <td style="padding:0 40px 16px 40px;">
              <p style="margin:0;font-size:16px;font-weight:700;color:#0f766e;">
                How it works:
              </p>
            </td>
          </tr>

          <!-- Feature 1: Upload -->
          <tr>
            <td style="padding:0 40px 12px 40px;">
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f0fdf4;border-radius:12px;border:1px solid #d1fae5;">
                <tr>
                  <td style="padding:20px 24px;">
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                      <tr>
                        <td width="44" valign="top">
                          <div style="width:40px;height:40px;background:linear-gradient(135deg,#0d9488,#14b8a6);border-radius:10px;text-align:center;line-height:40px;font-size:18px;">
                            &#128196;
                          </div>
                        </td>
                        <td style="padding-left:16px;">
                          <p style="margin:0 0 4px 0;font-size:14px;font-weight:700;color:#134e4a;">Upload records</p>
                          <p style="margin:0;font-size:13px;line-height:1.6;color:#64748b;">Lab tests, prescriptions, scans &ndash; anything.</p>
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
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f0fdf4;border-radius:12px;border:1px solid #d1fae5;">
                <tr>
                  <td style="padding:20px 24px;">
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                      <tr>
                        <td width="44" valign="top">
                          <div style="width:40px;height:40px;background:linear-gradient(135deg,#0d9488,#14b8a6);border-radius:10px;text-align:center;line-height:40px;font-size:18px;">
                            &#129504;
                          </div>
                        </td>
                        <td style="padding-left:16px;">
                          <p style="margin:0 0 4px 0;font-size:14px;font-weight:700;color:#134e4a;">Get AI analysis</p>
                          <p style="margin:0;font-size:13px;line-height:1.6;color:#64748b;">Understand what your results mean.</p>
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
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f0fdf4;border-radius:12px;border:1px solid #d1fae5;">
                <tr>
                  <td style="padding:20px 24px;">
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                      <tr>
                        <td width="44" valign="top">
                          <div style="width:40px;height:40px;background:linear-gradient(135deg,#0d9488,#14b8a6);border-radius:10px;text-align:center;line-height:40px;font-size:18px;">
                            &#128274;
                          </div>
                        </td>
                        <td style="padding-left:16px;">
                          <p style="margin:0 0 4px 0;font-size:14px;font-weight:700;color:#134e4a;">Share secure links</p>
                          <p style="margin:0;font-size:13px;line-height:1.6;color:#64748b;">With doctors or family when needed.</p>
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
            <td style="padding:0 40px 8px 40px;">
              <p style="margin:0;font-size:16px;font-weight:700;color:#1e293b;">
                Ready to start?
              </p>
            </td>
          </tr>

          <tr>
            <td style="padding:12px 40px 8px 40px;text-align:center;">
              <a href="https://app.onkeep.co/"
                 target="_blank"
                 style="display:inline-block;background:linear-gradient(135deg,#0d9488,#0f766e);color:#ffffff;text-decoration:none;padding:14px 36px;border-radius:10px;font-size:15px;font-weight:600;letter-spacing:0.02em;box-shadow:0 4px 14px rgba(13,148,136,0.35);">
                Upload your first record &rarr;
              </a>
            </td>
          </tr>

          <tr>
            <td style="padding:8px 40px 36px 40px;text-align:center;">
              <p style="margin:0;font-size:14px;color:#64748b;">
                &hellip;and see how simple it is.
              </p>
            </td>
          </tr>

          <!-- ============ DIVIDER ============ -->
          <tr>
            <td style="padding:0 40px;">
              <div style="height:1px;background-color:#d1fae5;"></div>
            </td>
          </tr>

          <!-- ============ FOOTER ============ -->
          <tr>
            <td style="padding:28px 40px 36px 40px;text-align:center;">
              <p style="margin:0 0 20px 0;font-size:14px;font-weight:700;color:#0f766e;">
                KEEP Team
              </p>
              <p style="margin:0;font-size:11px;color:#94a3b8;line-height:1.6;">
                &copy; {_get_current_year()} KEEP Health&ensp;|&ensp;
                <a href="https://onkeep.co" style="color:#0d9488;text-decoration:underline;">onkeep.co</a>
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
