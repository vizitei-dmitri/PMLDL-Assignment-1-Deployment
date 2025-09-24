import os
import requests
import streamlit as st
from datetime import datetime

API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Bank Marketing — Prediction", page_icon=":)", layout="centered")

st.markdown("""
<style>
.section {margin: 1rem 0 0.75rem 0; padding-top: .25rem;}
.hr {border:none; border-top:1px solid #2b2b2b; margin:.6rem 0 1rem 0}

.badge{display:inline-block;padding:.42rem .7rem;border-radius:.6rem;font-weight:600}
.badge-yes{background:#1b5e20;color:#fff}
.badge-no {background:#b71c1c;color:#fff}

.muted{color:#9aa0a6; font-size:.92rem}
</style>
""", unsafe_allow_html=True)

st.title("Deposit Subscription Prediction")

with st.form("predict_form", clear_on_submit=False):
    c1, c2 = st.columns(2, gap="large")

    with c1:
        age = st.number_input("age", 18, 95, 31)
        job = st.selectbox("job",
            ["admin.","blue-collar","entrepreneur","housemaid","management",
             "retired","self-employed","services","student","technician",
             "unemployed","unknown"])
        marital = st.selectbox("marital", ["married","single","divorced","unknown"])
        education = st.selectbox("education", ["primary","secondary","tertiary","unknown"])
        balance = st.number_input("balance", value=1000.0, step=50.0, format="%.2f")

    with c2:
        housing = st.checkbox("housing (mortgage)", value=True)
        loan = st.checkbox("loan (personal)", value=False)
        contact = st.selectbox("contact", ["cellular","telephone","unknown"])
        month = st.selectbox("month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
        campaign = st.number_input("campaign", 1, 30, 1)

    threshold = st.slider("threshold (decision for YES)", 0.05, 0.95, 0.50, 0.05)
    col_btn_l, col_btn_r = st.columns([1,1])
    submitted = col_btn_l.form_submit_button("Predict")
    reset     = col_btn_r.form_submit_button("Reset")

# if reset:
#     for k in list(st.session_state.keys()):
#         if k.startswith("predict_form"):  
#             st.session_state.pop(k)
#     st.experimental_rerun()


if reset:
    for k in ["age","job","marital","education","balance","housing","loan",
              "contact","month","campaign","threshold"]:
        st.session_state.pop(k, None)

    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()



if submitted:
    payload = {
        "age": age, "job": job, "marital": marital, "education": education,
        "balance": balance, "housing": housing, "loan": loan, "contact": contact,
        "month": month, "campaign": campaign
    }

    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

    if not r.ok:
        st.error(f"API error {r.status_code}: {r.text}")
        st.stop()

    data = r.json()
    proba = data.get("proba_yes", None)

    st.markdown("<div class='section'></div>", unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("Result")

    if proba is not None:
        st.progress(min(max(proba, 0.0), 1.0))
        st.write(f"Probability YES: **{proba:.1%}**")

        pred = 1 if proba >= threshold else 0
    else:
        pred = int(data.get("prediction", 0))
    st.markdown(
        '<span class="badge {}">Prediction: {} ({})</span>'.format(
            "badge-yes" if pred == 1 else "badge-no",
            "YES" if pred == 1 else "NO",
            pred,
        ),
        unsafe_allow_html=True,
    )
    st.caption(f"threshold = {threshold:.2f}")


    hist = st.session_state.get("history", [])
    hist.insert(0, {
        "ts": datetime.now().strftime("%H:%M:%S"),
        "payload": payload, "proba": proba, "pred": pred, "thr": float(threshold)
    })
    st.session_state["history"] = hist[:10]


if "history" in st.session_state and st.session_state["history"]:
    st.markdown("<div class='section'></div>", unsafe_allow_html=True)
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.subheader("History (last 10)")
    for i, h in enumerate(st.session_state["history"], 1):
        s = h["payload"]
        line = (
            f"{i}. {h['ts']} | pred={h['pred']}  proba={h['proba']:.3f}  thr={h['thr']:.2f}  |  "
            f"age={s['age']}, job={s['job']}, marital={s['marital']}, edu={s['education']}, "
            f"bal={s['balance']}, mortg={s['housing']}, loan={s['loan']}, "
            f"contact={s['contact']}, month={s['month']}, camp={s['campaign']}"
        )
        st.markdown(f"<span class='muted'>{line}</span>", unsafe_allow_html=True)

st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.subheader("Legend")
st.markdown("""
- **age** — возраст (18–95)  
- **job** — тип занятости (`retired`, `management`, `blue-collar`, …)  
- **marital** — семейное положение  
- **education** — образование  
- **balance** — средний баланс (может быть отрицательным ;) )  
- **housing** — ипотека (mortgage)
- **loan** — потребительский кредит (personal)
- **contact** — канал связи (`cellular`/`telephone`/`unknown`)  
- **month** — месяц контакта  
- **campaign** — число контактов в текущей кампании (1–30)  
""")
