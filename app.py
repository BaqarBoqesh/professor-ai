import os
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st

load_dotenv()

@st.cache_resource
def load_resources():
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return supabase, model, groq_client

supabase, model, groq_client = load_resources()

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

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"""당신은 교수님의 강의와 글을 기반으로 답변하는 AI입니다.

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
{context}"""},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content, docs

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
            answer, sources = ask(prompt)
            st.write(answer)
            if sources:
                with st.expander("📎 참고 자료"):
                    for doc in sources:
                        st.caption(doc["content"][:200] + "...")
    st.session_state.messages.append({"role": "assistant", "content": answer})