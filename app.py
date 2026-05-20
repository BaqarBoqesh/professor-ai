import os
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
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
    claude_client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")
    )
    return supabase, model, claude_client

supabase, model, claude_client = load_resources()

FILE_LIST_KEYWORDS = ["파일 목록", "어떤 자료", "몇 개 파일", "자료 목록", "파일 있어", "자료 있어", "어떤 파일"]

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

def search(query, limit=5):
    try:
        embedding = model.encode(query).tolist()
        result = supabase.rpc("match_documents", {
            "query_embedding": embedding,
            "match_count": limit
        }).execute()
        return result.data if result.data else []
    except Exception as e:
        st.error(f"검색 오류: {e}")
        return []

def ask(query):
    docs = search(query)
    if docs:
        context = "\n\n".join([d["content"] for d in docs])
        source_info = "아래 자료를 참고해서 답변하세요."
    else:
        context = "관련 자료를 찾지 못했습니다."
        source_info = "자료가 없으면 일반 지식으로 답변하세요."

    response = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=f"""당신은 교수님의 강의와 글을 기반으로 답변하는 AI입니다.

[절대 규칙]
1. 오직 한국어로만 답변하세요
2. 한자(漢字) 사용 금지
3. 일본어 사용 금지
4. 중국어 사용 금지
5. 러시아어 사용 금지
6. 모든 외국어 사용 금지
7. 고유명사도 반드시 한글로 표기 (예: 아도니베섹, 베섹)
8. 답변 전체를 한글로만 작성했는지 검토 후 출력하세요

{source_info}

참고 자료:
{context}""",
        messages=[
            {"role": "user", "content": query}
        ]
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