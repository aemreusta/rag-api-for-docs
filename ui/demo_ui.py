"""
Production-ready Streamlit Demo UI for Hürriyet Partisi AI Gateway.

This module provides a secure, user-friendly interface for interacting with
the chatbot API, including model selection, rate limiting, and comprehensive
error handling.
"""

import os
import time
import uuid

import requests
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Hürriyet Partisi AI Gateway",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🚀 Hürriyet Partisi AI Gateway")
st.markdown("*Parti politikaları hakkında AI destekli soru-cevap sistemi*")

# --- Session State Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rate_limit_info" not in st.session_state:
    st.session_state.rate_limit_info = {"limit": 0, "remaining": 0, "reset": 0}
if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = "Gemini Pro (OpenRouter)"
if "selected_model_id" not in st.session_state:
    st.session_state.selected_model_id = "google/gemini-1.5-pro-latest"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 1000

# --- Configuration (Securely from Environment) ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")

# Validate configuration
if not API_KEY:
    st.error("🚨 **FATAL:** API_KEY environment variable not set. The application cannot start.")
    st.info("""
    **Configuration Issue:** The API key is missing from the environment.

    **For Development:**
    1. Ensure your `.env` file contains `API_KEY=your_actual_key`
    2. Restart the containers: `make down && make up`

    **For Production:**
    1. Set the `API_KEY` environment variable in your deployment
    2. Restart the service
    """)
    st.stop()

# API endpoints
CHAT_ENDPOINT = f"{API_BASE_URL}/api/v1/chat"
RATE_LIMIT_ENDPOINT = f"{API_BASE_URL}/api/v1/rate-limit/status"

# Model configuration
MODEL_OPTIONS = {
    "Gemini Pro (OpenRouter)": "google/gemini-1.5-pro-latest",
    "Llama 3 70B (Groq)": "llama3-70b-8192",
    "ChatGPT-4o (OpenAI)": "gpt-4o",
}


# --- Helper Functions ---
def update_rate_limit_status() -> None:
    """
    Polls the rate limit status endpoint and updates session state.
    Uses the centrally configured API key from environment.
    """
    try:
        headers = {"X-API-Key": API_KEY}
        response = requests.get(RATE_LIMIT_ENDPOINT, headers=headers, timeout=5)
        if response.status_code == 200:
            st.session_state.rate_limit_info = response.json()
        elif response.status_code == 401:
            st.error("🔐 API anahtarı geçersiz. Lütfen .env dosyasını kontrol edin.")
        elif response.status_code == 422:
            st.error("🔐 API anahtarı eksik. Lütfen .env dosyasını kontrol edin.")
    except requests.RequestException:
        # Fail silently for rate limit status checks to avoid UI clutter
        pass


def get_chatbot_response(question: str, model: str) -> requests.Response | None:
    """
    Calls the backend with robust error handling.

    Args:
        question: User's question
        model: Selected model identifier

    Returns:
        Response object if successful, None otherwise
    """
    headers = {"X-API-Key": API_KEY}
    payload = {"question": question, "session_id": st.session_state.session_id, "model": model}

    try:
        response = requests.post(CHAT_ENDPOINT, json=payload, headers=headers, timeout=90)
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        error_msg = f"HTTP Hata {status_code}: "

        try:
            error_detail = e.response.json().get("detail", e.response.text)
        except requests.exceptions.JSONDecodeError:
            error_detail = e.response.text

        if status_code == 401:
            st.error(f"{error_msg}Geçersiz API anahtarı. Lütfen .env dosyasını kontrol edin.")
        elif status_code == 422:
            st.error(f"{error_msg}Geçersiz istek. {error_detail}")
        elif status_code == 429:
            st.warning(f"{error_msg}Hız sınırı aşıldı. Lütfen daha sonra tekrar deneyin.")
            # Automatically update rate limit info after 429
            update_rate_limit_status()
        elif status_code >= 500:
            st.error(f"{error_msg}Sunucu hatası. Lütfen daha sonra tekrar deneyin.")
        else:
            st.error(f"{error_msg}{error_detail}")

        return None
    except requests.exceptions.RequestException:
        st.error(
            f"🔗 Bağlantı Hatası: API'ye ulaşılamıyor ({CHAT_ENDPOINT}). Backend çalışıyor mu?"
        )
        return None


def format_sources(sources: list[dict]) -> None:
    """
    Display source information in a user-friendly format.

    Args:
        sources: List of source dictionaries from API response
    """
    if not sources:
        return

    with st.expander(f"📚 Kaynaklar ({len(sources)} belge)", expanded=False):
        for i, source in enumerate(sources, 1):
            score = source.get("score", 0.0)
            source_name = source.get("source", "N/A")
            page_num = source.get("page_number", "N/A")
            text = source.get("text", "N/A")

            # Color-code relevance score
            if score >= 0.8:
                score_color = "🟢"
            elif score >= 0.6:
                score_color = "🟡"
            else:
                score_color = "🔴"

            st.markdown(f"""
            **{i}. {source_name}** {score_color} (Sayfa {page_num}, Uygunluk: {score:.2f})
            *{text[:300]}{"..." if len(text) > 300 else ""}*
            """)


def format_time_remaining(reset_timestamp: int) -> str:
    """
    Format time remaining until rate limit reset.

    Args:
        reset_timestamp: Unix timestamp when limit resets

    Returns:
        Human-readable time string
    """
    remaining_seconds = max(0, reset_timestamp - int(time.time()))

    if remaining_seconds == 0:
        return "Şimdi sıfırlanıyor"
    elif remaining_seconds < 60:
        return f"{remaining_seconds}s"
    elif remaining_seconds < 3600:
        minutes = remaining_seconds // 60
        return f"{minutes}dk"
    else:
        hours = remaining_seconds // 3600
        return f"{hours}sa"


# --- Clean Sidebar (minimal brand only) ---
with st.sidebar:
    st.markdown("## 🏛️")
    st.caption("Hürriyet Partisi AI Gateway")

# --- Two-Column Layout: Chat (Left) | Info (Right) ---
col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.subheader("💬 Sohbet")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and "sources" in message:
                format_sources(message["sources"])

    # Chat input
    if prompt := st.chat_input("Parti politikaları hakkında soru sorun..."):
        if (
            st.session_state.rate_limit_info.get("remaining", 1) <= 0
            and st.session_state.rate_limit_info.get("limit", 0) > 0
        ):
            st.error("🚫 Hız sınırınız doldu. Lütfen sıfırlanmasını bekleyin.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🤔 Düşünüyor..."):
                    response = get_chatbot_response(prompt, st.session_state.selected_model_id)

                if response is not None:
                    try:
                        data = response.json()
                        answer = data.get("answer")
                        sources = data.get("sources", [])

                        if not answer:
                            answer = (
                                "Şu anda yanıt veremiyorum, ancak sorunu anladım. "
                                "Lütfen daha kısa veya farklı bir şekilde tekrar deneyin."
                            )

                        st.markdown(answer)
                        format_sources(sources)

                        assistant_message = {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                        st.session_state.messages.append(assistant_message)

                        update_rate_limit_status()

                    except requests.exceptions.JSONDecodeError:
                        fallback = (
                            "Beklenmeyen bir biçimde yanıt aldım. "
                            "Lütfen tekrar deneyin veya farklı bir soru sorun."
                        )
                        st.markdown(fallback)
                        st.session_state.messages.append({"role": "assistant", "content": fallback})
                else:
                    fallback = (
                        "Geçici bir bağlantı sorunu yaşanıyor. "
                        "Lütfen bağlantınızı kontrol edin ve tekrar deneyin."
                    )
                    st.markdown(fallback)
                    st.session_state.messages.append({"role": "assistant", "content": fallback})

with col_right:
    # Page-like selector; default shows 'Yapılandırma Bilgileri'
    info_page = st.radio(
        "Bilgi Panelleri",
        ["⚙️ Yapılandırma Bilgileri", "Ayarlar", "Durum"],
        index=0,
    )

    if info_page == "⚙️ Yapılandırma Bilgileri":
        st.subheader("⚙️ Yapılandırma Bilgileri")
        st.json(
            {
                "session_id": st.session_state.session_id,
                "message_count": len(st.session_state.messages),
                "rate_limit_info": st.session_state.rate_limit_info,
                "selected_model": st.session_state.selected_model_id,
                "api_base_url": API_BASE_URL,
                "config_source": "Environment Variables (.env)",
                "api_key_configured": bool(API_KEY),
            }
        )

    elif info_page == "Ayarlar":
        st.subheader("Ayarlar")
        selected_model_name = st.selectbox(
            "🤖 Model Seçimi",
            options=list(MODEL_OPTIONS.keys()),
            index=list(MODEL_OPTIONS.keys()).index(st.session_state.selected_model_name),
            help="Kullanılacak AI modelini seçin",
        )
        st.session_state.selected_model_name = selected_model_name
        st.session_state.selected_model_id = MODEL_OPTIONS[selected_model_name]

        st.subheader("🎛️ Model Parametreleri")
        st.session_state.temperature = st.slider(
            "Yaratıcılık (Temperature)",
            0.0,
            2.0,
            st.session_state.temperature,
            0.1,
            help="Yüksek değerler daha yaratıcı, düşük değerler daha tutarlı yanıtlar üretir",
        )
        st.session_state.max_tokens = st.slider(
            "Maksimum Kelime",
            100,
            4000,
            st.session_state.max_tokens,
            100,
            help="Yanıtın maksimum uzunluğu",
        )

        st.info(
            "💡 Model parametreleri şu anda yalnızca gösterge amaçlıdır. "
            "Gelecekteki sürümlerde aktif olacaktır."
        )

    elif info_page == "Durum":
        st.subheader("📱 Oturum Bilgileri")
        st.caption(f"**Oturum ID:** `{st.session_state.session_id[:8]}...`")
        if st.button("🔄 Oturumu Sıfırla", help="Yeni bir oturum başlatır ve geçmişi temizler"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.rate_limit_info = {"limit": 0, "remaining": 0, "reset": 0}
            st.success("✅ Oturum sıfırlandı!")
            st.rerun()

        st.divider()
        st.subheader("🚦 Hız Sınırı Durumu")
        rate_limit_container = st.container()

        if st.session_state.rate_limit_info["limit"] == 0:
            update_rate_limit_status()
        if st.button("🔄 Durumu Yenile", help="Hız sınırı durumunu günceller"):
            update_rate_limit_status()

        info = st.session_state.rate_limit_info
        if info["limit"] > 0:
            remaining_pct = (info["remaining"] / info["limit"]) * 100
            time_remaining = format_time_remaining(info["reset"])

            rate_limit_container.progress(remaining_pct / 100)
            rate_limit_container.info(
                f"**Kalan İstek:** {info['remaining']}/{info['limit']}  \n"
                f"**Sıfırlanma:** {time_remaining}"
            )

            if info["remaining"] <= 5 and info["remaining"] > 0:
                st.warning(f"⚠️ Sadece {info['remaining']} isteğiniz kaldı!")
            elif info["remaining"] == 0:
                st.error("🚫 Hız sınırı aşıldı. Lütfen bekleyin.")
        else:
            rate_limit_container.info("🔍 Hız sınırı durumu yükleniyor...")

# --- Footer ---
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🏛️ **Hürriyet Partisi AI Gateway**")
with col2:
    st.caption(f"🤖 **Aktif Model:** {st.session_state.selected_model_name}")
with col3:
    st.caption(f"🔗 **API:** {API_BASE_URL}")
