import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


AUTH_CSS = """
<style>
    .stApp {
        background-color: #f4f8f6;
        background-image: linear-gradient(rgba(24, 75, 67, .035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(24, 75, 67, .035) 1px, transparent 1px);
        background-size: 28px 28px;
    }
    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .nm-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        color: #173e38;
        font-size: 1.1rem;
        font-weight: 750;
    }
    .nm-mark {
        display: grid;
        width: 38px;
        height: 38px;
        place-items: center;
        border-radius: 11px;
        background: #176b5c;
        color: white;
        font-size: 1.1rem;
    }
    .nm-kicker {
        margin-top: 3.5rem;
        color: #b05b40;
        font-size: .78rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .nm-headline {
        max-width: 520px;
        margin: .8rem 0;
        color: #173e38;
        font-size: clamp(2.2rem, 5vw, 3.5rem);
        font-weight: 750;
        line-height: 1.04;
    }
    .nm-copy {
        max-width: 470px;
        color: #526863;
        font-size: 1rem;
        line-height: 1.65;
    }
    .nm-status {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 2.3rem;
        color: #46635c;
        font-size: .9rem;
    }
    .nm-dot {
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #d27755;
        box-shadow: 0 0 0 4px rgba(210, 119, 85, .14);
    }
    .nm-form-title {
        margin: 0 0 .25rem;
        color: #173e38;
        font-size: 1.65rem;
        font-weight: 700;
    }
    .nm-form-copy {
        margin-bottom: 1.1rem;
        color: #62746f;
        font-size: .94rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #dce7e2;
        border-radius: 8px;
        background: rgba(255, 255, 255, .94);
    }
    div.stButton > button, div.stLinkButton > a {
        min-height: 2.75rem;
        border-radius: 6px;
        font-weight: 650;
    }
    @media (max-width: 700px) {
        .nm-kicker { margin-top: 1.6rem; }
        .nm-headline { font-size: 2.35rem; }
    }
</style>
"""


def render_auth_gate() -> bool:
    """Show the NeuroMedix auth screens and return true for authenticated users."""
    if st.session_state.get("nm_authenticated"):
        return True

    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="nm-brand"><span class="nm-mark">N</span>'
        "NeuroMedix</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<div class="nm-kicker">NeuroMedix research workspace</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="nm-headline">Clinical insight, with care.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="nm-copy">A secure workspace for authorized teams to '
            "review research data and patient trends. Access is limited to "
            "approved accounts.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="nm-status"><span class="nm-dot"></span> '
            "Research prototype · Not for clinical decisions</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.write("")
        with st.container(border=True):
            view = st.radio(
                "Account access",
                ["Sign in", "Sign up"],
                horizontal=True,
                label_visibility="collapsed",
                key="nm_auth_view",
            )

            try:
                firebase_api_key = st.secrets.get("firebase_api_key", "")
            except StreamlitSecretNotFoundError:
                firebase_api_key = ""

            if view == "Sign in":
                st.markdown('<div class="nm-form-title">Welcome back</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="nm-form-copy">Sign in to your NeuroMedix '
                    "research workspace.</div>",
                    unsafe_allow_html=True,
                )
                with st.form("nm_sign_in", clear_on_submit=False):
                    email = st.text_input("Email address", autocomplete="email")
                    password = st.text_input(
                        "Password", type="password", autocomplete="current-password"
                    )
                    submitted = st.form_submit_button(
                        "Sign in", type="primary", use_container_width=True
                    )

                if submitted:
                    if not firebase_api_key:
                        st.error("Authentication is not configured. Add firebase_api_key to Streamlit secrets.")
                    elif not email or not password:
                        st.error("Enter both your email address and password.")
                    else:
                        try:
                            response = requests.post(
                                "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword",
                                params={"key": firebase_api_key},
                                json={"email": email, "password": password, "returnSecureToken": True},
                                timeout=15,
                            )
                            if response.ok:
                                account = response.json()
                                st.session_state["nm_authenticated"] = True
                                st.session_state["nm_email"] = account["email"]
                                st.rerun()
                            else:
                                st.error(_firebase_error(response))
                        except requests.RequestException:
                            st.error("Could not reach the authentication service. Try again.")

            elif view == "Sign up":
                st.markdown('<div class="nm-form-title">Create an account</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="nm-form-copy">Create credentials for your '
                    "NeuroMedix workspace.</div>",
                    unsafe_allow_html=True,
                )
                with st.form("nm_sign_up", clear_on_submit=False):
                    email = st.text_input("Email address", autocomplete="email")
                    password = st.text_input(
                        "Create password", type="password", autocomplete="new-password"
                    )
                    confirm_password = st.text_input(
                        "Confirm password", type="password", autocomplete="new-password"
                    )
                    submitted = st.form_submit_button(
                        "Create account", type="primary", use_container_width=True
                    )

                if submitted:
                    if not firebase_api_key:
                        st.error("Authentication is not configured. Add firebase_api_key to Streamlit secrets.")
                    elif not email or not password:
                        st.error("Enter an email address and password.")
                    elif password != confirm_password:
                        st.error("The passwords do not match.")
                    elif len(password) < 6:
                        st.error("Use a password with at least 6 characters.")
                    else:
                        try:
                            response = requests.post(
                                "https://identitytoolkit.googleapis.com/v1/accounts:signUp",
                                params={"key": firebase_api_key},
                                json={"email": email, "password": password, "returnSecureToken": True},
                                timeout=15,
                            )
                            if response.ok:
                                account = response.json()
                                st.session_state["nm_authenticated"] = True
                                st.session_state["nm_email"] = account["email"]
                                st.rerun()
                            else:
                                st.error(_firebase_error(response))
                        except requests.RequestException:
                            st.error("Could not reach the authentication service. Try again.")

            st.divider()
            st.caption(
                "Research prototype only. This application is not clinically "
                "validated and must not be used for care decisions."
            )

    return False


def _firebase_error(response: requests.Response) -> str:
    try:
        code = response.json().get("error", {}).get("message", "")
    except ValueError:
        code = ""

    messages = {
        "EMAIL_EXISTS": "An account with this email already exists. Sign in instead.",
        "INVALID_EMAIL": "Enter a valid email address.",
        "INVALID_LOGIN_CREDENTIALS": "The email or password is incorrect.",
        "WEAK_PASSWORD": "Choose a stronger password with at least 6 characters.",
        "TOO_MANY_ATTEMPTS_TRY_LATER": "Too many attempts. Wait a while and try again.",
        "OPERATION_NOT_ALLOWED": "Enable email/password sign-in in Firebase Authentication settings.",
    }
    return messages.get(code, "Authentication failed. Check the account details and try again.")