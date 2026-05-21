-- =====================================================
-- 하이브리드 검색 설정 (벡터 유사도 + 키워드 trigram)
-- Supabase SQL Editor에서 순서대로 실행하세요.
-- =====================================================


-- 1. pg_trgm 확장 활성화
-- trigram 기반 키워드 유사도 검색을 위해 필요한 PostgreSQL 확장
CREATE EXTENSION IF NOT EXISTS pg_trgm;


-- 2. content 컬럼에 GIN 인덱스 생성
-- trigram 검색 속도를 높이기 위한 인덱스 (full-text 키워드 매칭에 사용)
CREATE INDEX IF NOT EXISTS idx_documents_content_trgm
    ON documents
    USING GIN (content gin_trgm_ops);


-- 3. 하이브리드 검색 RPC 함수 생성
-- 벡터 유사도(pgvector)와 trigram 키워드 유사도를 RRF 방식으로 결합하여 반환
CREATE OR REPLACE FUNCTION match_documents_hybrid(
    query_embedding vector(768),  -- 질문 문장의 임베딩 벡터
    query_text      text,         -- 키워드 검색용 원문 텍스트
    match_count     int DEFAULT 10 -- 반환할 최대 결과 수
)
RETURNS TABLE (
    id         bigint,
    content    text,
    metadata   jsonb,
    similarity float
)
LANGUAGE plpgsql
STABLE
AS $$
BEGIN
    -- 한국어 trigram 유사도 임계값 조정
    -- 기본값 0.3은 한국어에 너무 높아 keyword_ranked가 항상 비게 됨 → 0.1로 완화
    SET LOCAL pg_trgm.similarity_threshold = 0.1;

    RETURN QUERY
    -- ① 벡터 유사도 기준 상위 후보 추출 및 순위 부여
    WITH vector_ranked AS (
        SELECT
            d.id,
            ROW_NUMBER() OVER (ORDER BY d.embedding <#> query_embedding) AS rank
        FROM documents d
        ORDER BY d.embedding <#> query_embedding
        LIMIT 60  -- RRF 안정성을 위해 상위 60개까지 후보 확보
    ),

    -- ② trigram 키워드 유사도 기준 상위 후보 추출 및 순위 부여
    keyword_ranked AS (
        SELECT
            d.id,
            ROW_NUMBER() OVER (ORDER BY similarity(d.content, query_text) DESC) AS rank
        FROM documents d
        WHERE d.content % query_text  -- trigram 유사도 임계값 이상인 문서만 대상
        ORDER BY similarity(d.content, query_text) DESC
        LIMIT 60
    ),

    -- ③ RRF(Reciprocal Rank Fusion)로 두 점수 결합
    -- 공식: score = 1/(60 + vector_rank) + 1/(60 + keyword_rank)
    -- 한쪽 결과에만 있는 문서도 포함 (FULL OUTER JOIN)
    rrf_scored AS (
        SELECT
            COALESCE(v.id, k.id) AS id,
            COALESCE(1.0 / (60 + v.rank), 0)::float
            + COALESCE(1.0 / (60 + k.rank), 0)::float AS rrf_score
        FROM vector_ranked  v
        FULL OUTER JOIN keyword_ranked k ON v.id = k.id
    )

    -- ④ RRF 점수 상위 match_count개 문서 반환
    SELECT
        d.id,
        d.content,
        d.metadata,
        r.rrf_score AS similarity
    FROM rrf_scored r
    JOIN documents d ON d.id = r.id
    ORDER BY r.rrf_score DESC
    LIMIT match_count;

END;
$$;
