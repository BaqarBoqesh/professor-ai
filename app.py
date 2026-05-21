import os
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer, CrossEncoder
import anthropic

load_dotenv()

# 비밀번호 보호
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.title("🔐 AI 교수님")
        password = st.text_input("비밀번호를 입력하세요", type="password")
        if st.button("로그인"):
            if password == st.secrets["APP_PASSWORD"]:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("비밀번호가 틀렸습니다")
        st.stop()

check_password()

@st.cache_resource
def load_resources():
    supabase = create_client(
        os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL"),
        os.getenv("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")
    )
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    # Before: CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # 영어 전용 모델
    reranker = CrossEncoder("bongsoo/mmarco-mMiniLMv2-L12-H384-v1")  # 한국어 포함 다국어 모델
    claude_client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
    )
    return supabase, model, reranker, claude_client

supabase, model, reranker, claude_client = load_resources()

# 정적 시스템 프롬프트 — 요청마다 변하지 않으므로 모듈 상수로 정의해 캐시 효율 확보
_SYSTEM_STATIC = """당신은 교수님의 강의와 글을 기반으로 답변하는 AI입니다.

[절대 규칙]
1. 오직 한국어로만 답변하세요
2. 한자(漢字) 사용 금지
3. 일본어 사용 금지
4. 중국어 사용 금지
5. 러시아어 사용 금지
6. 모든 외국어 사용 금지
7. 고유명사도 반드시 한글로 표기 (예: 아도니베섹, 베섹)
8. 답변 전체를 한글로만 작성했는지 검토 후 출력하세요"""

FILE_LIST_KEYWORDS = ["파일 목록", "어떤 자료", "몇 개 파일", "자료 목록", "파일 있어", "자료 있어", "어떤 파일", "몇 개야", "올린 파일"]

def is_file_list_query(query):
    return any(kw in query for kw in FILE_LIST_KEYWORDS)

def get_file_list():
    try:
        result = supabase.table("documents").select("metadata").execute()
        if not result.data:
            return "등록된 자료가 없습니다."
        sources = sorted({
            row["metadata"]["source"]
            for row in result.data
            if row.get("metadata") and row["metadata"].get("source")
        })
        if not sources:
            return "파일 정보를 찾을 수 없습니다."
        lines = "\n".join(f"- {s}" for s in sources)
        return f"현재 등록된 자료는 총 {len(sources)}개입니다:\n\n{lines}"
    except Exception as e:
        return f"자료 목록 조회 오류: {e}"

@st.cache_data(ttl=300)
def get_known_speakers():
    try:
        result = supabase.table("documents").select("metadata").execute()
        speakers = set()
        for row in result.data:
            for s in (row.get("metadata") or {}).get("speakers") or []:
                speakers.add(s)
        return speakers
    except Exception:
        return set()

@st.cache_data(ttl=300)
def get_known_sources():
    try:
        result = supabase.table("documents").select("metadata").execute()
        sources = set()
        for row in result.data:
            src = (row.get("metadata") or {}).get("source")
            if src:
                sources.add(os.path.splitext(os.path.basename(src))[0])
        return sources
    except Exception:
        return set()

def extract_speaker_from_query(query):
    return next((s for s in get_known_speakers() if s in query), None)

def extract_source_from_query(query):
    return next((s for s in get_known_sources() if s in query), None)

def search(query, limit=5, speaker=None, source=None):
    try:
        embedding = model.encode(query).tolist()

        if speaker or source:
            # Before: 전체 테이블 조회 후 numpy 코사인 유사도 재계산
            # q_builder = supabase.table("documents").select("content, metadata, embedding")
            # if speaker:
            #     q_builder = q_builder.contains("metadata", {"speakers": [speaker]})
            # result = q_builder.execute()
            # data = result.data or []
            # if source:
            #     data = [row for row in data if source in (row.get("metadata") or {}).get("source", "")]
            # if data:
            #     q = np.array(embedding)
            #     def cosine(row):
            #         raw = row["embedding"]
            #         e = np.array(json.loads(raw) if isinstance(raw, str) else raw)
            #         return float(np.dot(q, e) / (np.linalg.norm(q) * np.linalg.norm(e) + 1e-9))
            #     return sorted(data, key=cosine, reverse=True)[:limit]
            # return []

            # 하이브리드 검색으로 상위 후보 확보 후 Python에서 메타데이터 필터 적용
            # (RPC가 메타데이터 필터를 지원하지 않으므로 충분한 후보를 받아 Python에서 걸러냄)
            result = supabase.rpc("match_documents_hybrid", {
                "query_embedding": embedding,
                "query_text": query,
                "match_count": 60,
            }).execute()
            data = result.data or []
            if speaker:
                data = [row for row in data if speaker in (row.get("metadata") or {}).get("speakers", [])]
            if source:
                data = [row for row in data if source in (row.get("metadata") or {}).get("source", "")]
            return data[:limit]

        # Before: supabase.rpc("match_documents", {"query_embedding": embedding, "match_count": limit})
        result = supabase.rpc("match_documents_hybrid", {
            "query_embedding": embedding,
            "query_text": query,
            "match_count": limit,
        }).execute()
        return result.data if result.data else []

    except Exception as e:
        st.error(f"검색 오류: {e}")
        return []

def rerank(query, docs, top_n=4):
    """CrossEncoder로 (질문, 청크) 쌍을 채점해 상위 top_n개만 반환."""
    if not docs or len(docs) <= top_n:
        return docs
    pairs = [(query, doc["content"]) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]

def ask(query):
    speaker = extract_speaker_from_query(query)
    source = extract_source_from_query(query)
    # Before: docs = search(query, speaker=speaker, source=source)
    docs = search(query, limit=10, speaker=speaker, source=source)
    docs = rerank(query, docs, top_n=4)
    if docs:
        context = "\n\n".join([d["content"] for d in docs])
        if source:
            source_info = f"'{source}' 자료에서 찾은 내용을 참고해서 답변하세요."
        else:
            source_info = "아래 자료를 참고해서 답변하세요."
    else:
        context = "관련 자료를 찾지 못했습니다."
        source_info = "자료가 없으면 일반 지식으로 답변하세요."

    # Before:
    # response = claude_client.messages.create(
    #     model="claude-haiku-4-5-20251001",
    #     max_tokens=2048,
    #     system=f"""당신은 교수님의 강의와 글을 기반으로 답변하는 AI입니다.
    #
    # [절대 규칙]
    # 1. 오직 한국어로만 답변하세요
    # 2. 한자(漢字) 사용 금지
    # 3. 일본어 사용 금지
    # 4. 중국어 사용 금지
    # 5. 러시아어 사용 금지
    # 6. 모든 외국어 사용 금지
    # 7. 고유명사도 반드시 한글로 표기 (예: 아도니베섹, 베섹)
    # 8. 답변 전체를 한글로만 작성했는지 검토 후 출력하세요
    #
    # {source_info}
    #
    # 참고 자료:
    # {context}""",
    #     messages=[{"role": "user", "content": query}]
    # )

    response = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_STATIC,
                "cache_control": {"type": "ephemeral"},  # 정적 규칙 캐시
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{source_info}\n\n참고 자료:\n{context}",
                        "cache_control": {"type": "ephemeral"},  # 검색 컨텍스트 캐시
                    },
                    {
                        "type": "text",
                        "text": query,
                    },
                ],
            }
        ],
    )
    return response.content[0].text, docs

st.title("📚 AI 교수님")
st.caption("교수님의 강의와 글을 기반으로 답변합니다")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            if is_file_list_query(prompt):
                answer = get_file_list()
                st.write(answer)
            else:
                answer, sources = ask(prompt)
                st.write(answer)
                if sources:
                    with st.expander("📎 참고 자료"):
                        for doc in sources:
                            st.caption(doc["content"][:200] + "...")
    st.session_state.messages.append({"role": "assistant", "content": answer})