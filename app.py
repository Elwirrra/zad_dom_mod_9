# === IMPORTY ===


import os, io, json, functools
import pandas as pd
import streamlit as st
import joblib, boto3
from pydantic import BaseModel, ValidationError, conint, confloat
from pathlib import Path
from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.openai import OpenAI as LangfuseOpenAI
import re


# === KONFIG ===


load_dotenv(Path(__file__).parent / ".env", override=True) 

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

client = LangfuseOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BUCKET = os.getenv("SPACES_BUCKET")                 
ENDPOINT = os.getenv("AWS_ENDPOINT_URL_S3")             
ACCESS = os.getenv("AWS_ACCESS_KEY_ID")
SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
MODEL_KEY = os.getenv("MODEL_KEY")                   

REQUIRED = ["plec", "wiek", "czas_5km"]           

ERROR_MSG_MISSING = (
    "Brak wystarczających danych. Upewnij się, że wprowadziłeś "
    "wiek, płeć i czas na 5km biegu!"
)

st.set_page_config(
    page_title="Predykcja czasu półmaratonu",
    page_icon="🏃‍♀️",
    layout="wide"
)


# === STYL NAGŁÓWKA ===


st.markdown("""
<style>
h1.app-title{
  font-size: 2.8rem; font-weight: 800; line-height: 1.1;
  letter-spacing:.2px; margin: .2rem 0 .4rem 0;
  background: linear-gradient(90deg,#0ea5e9 0%, #22c55e 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
p.app-subtitle{
  font-size: 2.18rem;    /* było ~0.95rem */
  font-weight: 600;
  color: #475569;
  margin: .25rem 0 .9rem 0;
}
h1.app-title, p.app-subtitle{
  text-align: center !important;
  margin-left: auto; 
  margin-right: auto;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="app-title">🏃‍♀️ Sprawdź jak szybki będziesz w półmaratonie?</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Podaj wiek, płeć i Twój czas na 5 km - resztę zrobię za Ciebie.</p>', unsafe_allow_html=True)


# === FUNKCJE POMOCNICZE ===


def lf_start_trace(lf, **kwargs):
    """Zacznij trace niezależnie od wersji SDK."""
    if not lf:
        return None
    for name in ("create_trace", "trace", "start_trace"):
        fn = getattr(lf, name, None)
        if callable(fn):
            return fn(**kwargs)
    return None

def lf_start_span(parent, **kwargs):
    """Zacznij span niezależnie od wersji SDK."""
    if not parent:
        return None
    for name in ("create_span", "span", "start_span"):
        fn = getattr(parent, name, None)
        if callable(fn):
            return fn(**kwargs)
    return None

def lf_end_span(span, **kwargs):
    """Zakończ span niezależnie od wersji SDK."""
    if not span:
        return
    # najczęściej .end(output=...), ale czasem .update(output=...) lub .finish()
    for name in ("end", "update", "finish", "close"):
        fn = getattr(span, name, None)
        if callable(fn):
            try:
                fn(**kwargs)
            except TypeError:
                # spróbuj z samym output=...
                out = kwargs.get("output")
                if out is not None:
                    try:
                        fn(output=out)
                    except Exception:
                        pass
            break


@st.cache_resource
def _lf():
    if Langfuse is None:
        return None
    try:
        return Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception:
         return None

def observe(name: str):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            lf = _lf()
            trace = lf_start_trace(lf, name=name)
            span  = lf_start_span(trace, name=name, input={"args": args, "kwargs": kwargs})
            try:
                result = fn(*args, **kwargs)
                lf_end_span(span, output=result)
                return result
            except Exception as e:
                lf_end_span(span, output={"error": str(e)})
                raise
        return wrapper
    return deco

@st.cache_resource
def get_langfuse():
    try:
        return Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except:
        return None

def s3():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS,
        aws_secret_access_key=SECRET,
    )

@st.cache_resource(show_spinner=False)
def load_model():
    s3c = s3()
    # szybki pre-check: czy plik istnieje?
    try:
        s3c.head_object(Bucket=BUCKET, Key=MODEL_KEY)
    except Exception as e:
        raise FileNotFoundError(
            f"[diag] Nie ma pliku modelu pod kluczem '{MODEL_KEY}' w buckecie '{BUCKET}'. "
            f"Sprawdź nazwę/katalog w Spaces. Błąd: {e}"
        )

    obj = s3c.get_object(Bucket=BUCKET, Key=MODEL_KEY)
    model = joblib.load(io.BytesIO(obj["Body"].read()))
    return model, MODEL_KEY

    
class Inputs(BaseModel):
    plec: str                # "K" lub "M"
    wiek: conint(ge=10, le=100)
    czas_5km: confloat(gt=10*60, lt=60*60)  # sekundy (10–60 min)


@observe(name="extract_with_llm")
def extract_with_llm(text: str) -> dict:

    # Twarde ograniczenia pomagają modelowi i odsieją bzdury
    schema = {
        "name": "inputs",
        "schema": {
            "type": "object",
            "properties": {
                "plec": {"type": "string", "enum": ["K","M", None]},
                "wiek": {"type": "integer", "minimum": 10, "maximum": 100},
                # czas w SEKUNDACH (10–60 minut → 600–3600 s)
                "czas_5km": {"type": "number", "minimum": 600, "maximum": 3600}
            },
            "required": ["plec","wiek","czas_5km"],
            "additionalProperties": False
        }
    }

    sys = """
    Wyodrębnij 'plec' (K/M), 'wiek' (w latach) oraz 'czas_5km' w SEKUNDACH.
    Użytkownik może podać czas w minutach (np. '33 min'), w formacie 'MM:SS' albo 'HH:MM:SS'.
    Zawsze przelicz na sekundy i zwróć WYŁĄCZNIE JSON zgodny ze schematem.
    Jeśli którejkolwiek informacji NIE podano wprost, zwróć wartość null (NIE zgaduj).

    MAPOWANIE PŁCI:
    - Jeśli pada 'mężczyzna', 'facet', 'chłopak', 'pan' → zwróć "M".
    - Jeśli pada 'kobieta', 'dziewczyna', 'pani' → zwróć "K".
    - Jeśli użytkownik poda imię męskie (np. 'Jestem Piotrek/Piotr/Paweł/Marcin/Adam') → "M".
    - Jeśli użytkownik poda imię żeńskie (np. 'Jestem Ania/Anna/Kasia/Joanna/Magda') → "K".
    - W razie niejednoznaczności → null.

    Przykłady:
    - 'kobieta, 34 lata, 5 km 27:30'            → {"plec":"K","wiek":34,"czas_5km":1650}
    - 'M 40 lat, 5 km w 33 min'                  → {"plec":"M","wiek":40,"czas_5km":1980}
    - 'Jestem Piotrek, 47 lat, 5 km 40 min'      → {"plec":"M","wiek":47,"czas_5km":2400}
    - 'Jestem Ania, 19 lat, 5km w 18m45s'        → {"plec":"K","wiek":19,"czas_5km":1125}
    - 'Mam 40 lat, 5 km w 25:10'                 → {"plec": null, "wiek":40,"czas_5km":1510}
    """
        
    

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),
        temperature=0,
        messages=[
            {"role":"system","content": sys},
            {"role":"user","content": text}
        ],
        response_format={"type":"json_schema","json_schema": schema}
    )
    return json.loads(resp.choices[0].message.content)


def to_features(p: Inputs) -> pd.DataFrame:
    pace_min_per_km = (p.czas_5km / 60.0) / 5.0   # sek→min, /5 km = min/km
    return pd.DataFrame([{
        "wiek": p.wiek,
        "plec": p.plec,
        "start_druzynowy": "nie",
        "5km_tempo": pace_min_per_km,
    }])

def fmt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    else:
        return f"{m}:{s:02d}"

def infer_gender_from_text(text: str) -> str | None:
    t = text.lower()
    male_kw = ["mężczyzna","mezczyzna","facet","chłopak","chlopak","pan"]
    female_kw = ["kobieta","dziewczyna","pani"]

    if any(k in t for k in male_kw): 
        return "M"
    if any(k in t for k in female_kw):
        return "K"

    # prosta heurystyka po imieniu: "jestem <imię>" lub "nazywam się <imię>"
    m = re.search(r"(jestem|nazywam się|nazywam sie)\s+([A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż\-]+)", text, flags=re.I)
    if m:
        name = m.group(2)
        # wyjątki imion męskich na -a
        male_exceptions = {"Kuba","Barnaba","Kosma","Bonawentura","Ilya","Sasza","Nikita"}
        if name in male_exceptions or not name.endswith("a"):
            return "M"
        else:
            return "K"
    return None         

# === UI ===


text = st.text_area(
    label="", 
    label_visibility="collapsed",
    placeholder="Np.: „Mam 34 lata, kobieta, 5 km w 27:30”.",
    height=120,
    key="user_text",
)




if st.button("Policz", key="policz_main"):
    lf = get_langfuse()
    trace = lf_start_trace(lf, name="predykcja", input={"raw_text": text})

    try:
        # DEBUG – pokaż, czy env vars są widoczne
        if DEBUG:
            st.info({
                "HAS_OPENAI_KEY": bool(os.getenv("OPENAI_API_KEY")),
                "HAS_LANGFUSE_PUB": bool(os.getenv("LANGFUSE_PUBLIC_KEY")),
                "HAS_LANGFUSE_SEC": bool(os.getenv("LANGFUSE_SECRET_KEY")),
                "BUCKET": os.getenv("SPACES_BUCKET"),
                "MODEL_KEY_env": os.getenv("MODEL_KEY"),
                "LATEST_KEY_env": os.getenv("LATEST_KEY"),
            })

        # 1) Parsowanie LLM
        data_raw = extract_with_llm(text)

        # DEBUG – pokaż co zwrócił LLM
        if DEBUG:
            st.code(f"LLM parsed: {json.dumps(data_raw, ensure_ascii=False)}", language="json")

        # 2) Heurystyka płci po LLM (opcjonalnie, jeśli dodałaś funkcję)
        plec = data_raw.get("plec")
        if plec not in {"K","M"}:
            guessed = infer_gender_from_text(text)
            if guessed:
                data_raw["plec"] = guessed

        # 3) Twarda walidacja
        for k in REQUIRED:
            if not data_raw.get(k):
                raise ValueError(f"missing:{k}")

        data = Inputs(**data_raw)

        # 4) Predykcja
        model, model_key = load_model()
        X = to_features(data)
        y_sec = float(model.predict(X)[0])

        st.success(f"Szacowany wynik: {y_sec/60:.1f} min ({fmt_time(y_sec)})")
        st.caption(f"Model: {model_key}")

        if trace:
            trace.update(output={"pred": y_sec, "model_key": model_key})
            lf.flush()

    except Exception as e:
        if DEBUG:
            st.exception(e)
        else:
            st.error("Nie udało się policzyć wyniku, sprawdź czy wszystkie potrzebne dane zostały wprowadzone.")
        if trace:
            trace.update(tags=["error"], output={"error": str(e)})
            lf.flush()
                
    # try:
    #     data_raw = extract_with_llm(text)
        
    #     plec = data_raw.get("plec")
    #     if plec not in {"K","M"}:
    #         guessed = infer_gender_from_text(text)
    #         if guessed:
    #             data_raw["plec"] = guessed
    #     for k in REQUIRED:
    #         if not data_raw.get(k):
    #             raise ValueError(f"missing:{k}")
    #     data = Inputs(**data_raw)

    #     model, model_key = load_model()
    #     X = to_features(data)
    #     y_sec = float(model.predict(X)[0])

    #     # --- wynik dla użytkownika ---
    #     st.success(f"Szacowany wynik: {y_sec/60:.1f} min ({fmt_time(y_sec)})")
    #     st.caption(f"Model: {model_key}")

    #     # --- zapis do Langfuse ---
    #     if trace:
    #         trace.update(output={"pred": y_sec, "model_key": model_key})
    #         lf.flush()

    # except Exception as e:
    #     st.error("Nie udało się policzyć wyniku, sprawdź czy wszystkie potrzebne dane zostały wprowadzone.")
    #     if trace:
    #         trace.update(tags=["error"], output={"error": str(e)})
    #         lf.flush()


