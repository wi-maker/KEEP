# KEEP Direct Google OAuth Setup Guide

To finish the transition to the direct Google OAuth flow (bypassing the `supabase.co` consent screen domain), please complete the following two steps.

## 1. Add Secrets to Railway (Backend)
The backend now securely handles the OAuth token exchange. It needs your Google Client ID and Secret.

1. Go to your **KEEP** project in [Railway](https://railway.app/).
2. Select your **Backend** service.
3. Go to the **Variables** tab.
4. Add the following two variables:
   - `GOOGLE_CLIENT_ID`: `1096901848529-g5m6k2o326774q5r9962aam4kcvk6052.apps.googleusercontent.com`
   - `GOOGLE_CLIENT_SECRET`: *(The secret associated with this client ID from Google Cloud Console)*
5. Wait for the backend to redeploy.

## 2. Update Authorized Redirect URIs in Google Cloud
Google needs to know that your Vercel/frontend domain is allowed to request logins.

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Navigate to **APIs & Services** > **Credentials**.
3. Edit your OAuth 2.0 Client ID (`1096901848529-g5m...`).
4. Under **Authorized redirect URIs**, add your exact production URL for the KEEP frontend, exactly as it appears in the `window.location.origin` (e.g., `https://app.onkeep.co` or `https://keep-admin.vercel.app`), without any trailing slashes or URL fragments.
   - **Example:** `https://app.onkeep.co`
   - *Note: Google strictly forbids URL fragments (like `#auth`) in redirect URIs.*
5. Save the changes.

## Testing
Once the backend has deployed and the Google Cloud Console is updated:
1. Go to your KEEP frontend (e.g., `https://app.onkeep.co`).
2. Click **Sign in with Google**.
3. You should be redirected straight to the Google consent screen (showing your custom domain, not Supabase).
4. After consenting, you will be redirected back to the KEEP frontend.
5. The frontend will automatically detect the `?code=` in the URL, exchange it securely with the backend, log you into Supabase, and refresh the UI.
