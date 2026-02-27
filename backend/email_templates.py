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
<body style="margin:0;padding:0;background-color:#F9FAFB;font-family:'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif;-webkit-font-smoothing:antialiased;">

  <!-- Outer wrapper -->
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#F9FAFB;">
    <tr>
      <td align="center" style="padding:40px 16px;">

        <!-- Email card -->
        <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;background-color:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 10px 25px rgba(0,0,0,0.05);border:1px solid #E5E7EB;">

          <!-- ============ HEADER ============ -->
          <tr>
            <td style="background-color:#79C412;padding:40px 40px 32px 40px;text-align:center;">
              <!-- Logo -->
              <img
                src="https://i.ibb.co/SX345WJg/photo-2026-02-18-14-07-39.jpg"
                alt="KEEP"
                width="120"
                style="display:inline-block;width:120px;height:auto;"
              />
              <p style="margin:16px 0 0 0;font-size:13px;letter-spacing:0.08em;text-transform:uppercase;color:#ffffff;font-weight:600;opacity:0.9;">
                Your Personal Health Vault
              </p>
            </td>
          </tr>

          <!-- ============ GREETING ============ -->
          <tr>
            <td style="padding:40px 40px 16px 40px;">
              <p style="margin:0;font-size:16px;line-height:1.7;color:#374151;">
                Hi {first_name},
              </p>
            </td>
          </tr>

          <!-- ============ HEADLINE ============ -->
          <tr>
            <td style="padding:0 40px 16px 40px;">
              <h1 style="margin:0;font-size:24px;font-weight:800;color:#111827;letter-spacing:-0.02em;">
                Welcome to KEEP &ndash; your personal health vault.
              </h1>
            </td>
          </tr>

          <!-- ============ INTRO ============ -->
          <tr>
            <td style="padding:0 40px 32px 40px;">
              <p style="margin:0;font-size:16px;line-height:1.6;color:#4B5563;">
                KEEP stores all your medical records in one secure place, so you no longer have to deal with lost results or scattered paperwork.
              </p>
            </td>
          </tr>

          <!-- ============ HOW IT WORKS HEADING ============ -->
          <tr>
            <td style="padding:0 40px 16px 40px;">
              <h2 style="margin:0;font-size:14px;font-weight:700;color:#6B7280;text-transform:uppercase;letter-spacing:0.05em;">
                How it works
              </h2>
            </td>
          </tr>

          <!-- Feature 1: Upload -->
          <tr>
            <td style="padding:0 40px 16px 40px;">
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#F9FAFB;border-radius:12px;border:1px solid #E5E7EB;">
                <tr>
                  <td style="padding:20px 24px;">
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                      <tr>
                        <td width="48" valign="top">
                          <div style="width:48px;height:48px;background-color:rgba(121,196,18,0.1);border-radius:12px;text-align:center;line-height:48px;font-size:20px;">
                            <span style="display:inline-block;vertical-align:middle;">&#128196;</span>
                          </div>
                        </td>
                        <td style="padding-left:16px;" valign="middle">
                          <p style="margin:0 0 4px 0;font-size:15px;font-weight:600;color:#111827;">Upload records</p>
                          <p style="margin:0;font-size:14px;line-height:1.5;color:#4B5563;">Lab tests, prescriptions, scans &ndash; anything.</p>
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
            <td style="padding:0 40px 16px 40px;">
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#F9FAFB;border-radius:12px;border:1px solid #E5E7EB;">
                <tr>
                  <td style="padding:20px 24px;">
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                      <tr>
                        <td width="48" valign="top">
                          <div style="width:48px;height:48px;background-color:rgba(121,196,18,0.1);border-radius:12px;text-align:center;line-height:48px;font-size:20px;">
                            <span style="display:inline-block;vertical-align:middle;">&#129504;</span>
                          </div>
                        </td>
                        <td style="padding-left:16px;" valign="middle">
                          <p style="margin:0 0 4px 0;font-size:15px;font-weight:600;color:#111827;">Get AI analysis</p>
                          <p style="margin:0;font-size:14px;line-height:1.5;color:#4B5563;">Understand what your results mean simply.</p>
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
            <td style="padding:0 40px 32px 40px;">
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#F9FAFB;border-radius:12px;border:1px solid #E5E7EB;">
                <tr>
                  <td style="padding:20px 24px;">
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                      <tr>
                        <td width="48" valign="top">
                          <div style="width:48px;height:48px;background-color:rgba(121,196,18,0.1);border-radius:12px;text-align:center;line-height:48px;font-size:20px;">
                            <span style="display:inline-block;vertical-align:middle;">&#128274;</span>
                          </div>
                        </td>
                        <td style="padding-left:16px;" valign="middle">
                          <p style="margin:0 0 4px 0;font-size:15px;font-weight:600;color:#111827;">Share secure links</p>
                          <p style="margin:0;font-size:14px;line-height:1.5;color:#4B5563;">With doctors or family when needed.</p>
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
            <td style="padding:0 40px 16px 40px;">
              <p style="margin:0;font-size:18px;font-weight:700;color:#111827;text-align:center;">
                Ready to start?
              </p>
            </td>
          </tr>

          <tr>
            <td style="padding:0 40px 12px 40px;text-align:center;">
              <a href="https://app.onkeep.co/"
                 target="_blank"
                 style="display:inline-block;background-color:#79C412;color:#ffffff;text-decoration:none;padding:16px 40px;border-radius:8px;font-size:15px;font-weight:600;letter-spacing:0.02em;box-shadow:0 4px 14px rgba(121,196,18,0.25);">
                Upload your first record
              </a>
            </td>
          </tr>

          <tr>
            <td style="padding:0 40px 40px 40px;text-align:center;">
              <p style="margin:0;font-size:14px;color:#6B7280;">
                &hellip;and see how simple it is.
              </p>
            </td>
          </tr>

          <!-- ============ DIVIDER ============ -->
          <tr>
            <td style="padding:0 40px;">
              <div style="height:1px;background-color:#E5E7EB;"></div>
            </td>
          </tr>

          <!-- ============ FOOTER ============ -->
          <tr>
            <td style="padding:32px 40px 40px 40px;text-align:center;">
              <p style="margin:0 0 16px 0;font-size:14px;font-weight:700;color:#111827;">
                KEEP Team
              </p>
              <p style="margin:0;font-size:12px;color:#6B7280;line-height:1.6;">
                &copy; {_get_current_year()} KEEP Health &bull; 
                <a href="https://onkeep.co" style="color:#79C412;text-decoration:underline;">onkeep.co</a>
              </p>
              <p style="margin:8px 0 0 0;font-size:12px;color:#9CA3AF;">
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
