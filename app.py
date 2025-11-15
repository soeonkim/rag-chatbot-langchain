import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 데이터 폴더 경로
DATA_DIR = "./data"

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

# 전역 변수: Lazy Loading을 위한 캐시
_embeddings_cache = None
_retriever_cache = None

def get_embeddings():
    """
    임베딩 모델을 Lazy Loading으로 반환
    - 최초 호출 시에만 초기화
    - 이후 호출 시 캐시된 객체 재사용
    """
    global _embeddings_cache

    if _embeddings_cache is not None:
        print("✅ 캐시된 임베딩 모델 사용")
        return _embeddings_cache

    print("🔧 임베딩 모델 최초 로딩 중...")
    _embeddings_cache = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )
    print("✅ 임베딩 모델 로드 완료 및 캐싱")
    return _embeddings_cache

def get_retriever():
    """
    Retriever를 Lazy Loading으로 반환
    - 최초 호출 시에만 초기화 (KB 미사용 시 리소스 낭비 없음)
    - 이후 호출 시 캐시된 객체 재사용 (빠른 응답)
    """
    global _retriever_cache

    # 이미 초기화된 경우 캐시된 객체 반환
    if _retriever_cache is not None:
        print("✅ 캐시된 Retriever 사용")
        return _retriever_cache

    # 최초 호출 시에만 초기화
    try:
        print("🔧 Retriever 최초 초기화 중...")
        embeddings = get_embeddings()  # 공통 함수 사용

        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db"
        )

        # 벡터DB가 비어있는지 확인
        try:
            doc_count = vector_store._collection.count()
            if doc_count == 0:
                print("⚠️ 벡터DB가 비어있습니다. '벡터DB 리로드'를 먼저 실행하세요.")
                return None
            print(f"📊 벡터DB에 {doc_count}개 문서 존재")
        except Exception as count_error:
            print(f"⚠️ 문서 개수 확인 실패: {count_error}")
            return None

        _retriever_cache = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )
        print("✅ Retriever 초기화 완료 및 캐싱")
        return _retriever_cache

    except Exception as e:
        print(f"❌ Retriever 초기화 실패: {e}")
        return None
    
llm = get_llm()

def format_docs(docs):
    """
    Chroma에서 검색된 문서를 문자열로 변환

    Args:
        docs: LangChain Document 객체 리스트

    Returns:
        str: 포맷된 문서 내용
    """
    if not docs:
        print("⚠️ 검색된 문서 없음")
        return "관련 문서를 찾을 수 없습니다."

    formatted = []
    for idx, doc in enumerate(docs, 1):
        # Chroma는 LangChain Document 객체를 반환
        content = doc.page_content.strip()

        # 메타데이터에서 출처 정보 추출
        source = doc.metadata.get("source", "알 수 없음")
        page = doc.metadata.get("page")

        # 문서 번호와 출처 정보 포함
        doc_info = f"[문서 {idx}] 출처: {source}"
        if page is not None:
            # page는 0부터 시작하므로 +1
            doc_info += f", 페이지: {page + 1}"

        formatted.append(f"{doc_info}\n{content}")

    result = "\n\n---\n\n".join(formatted)
    print(f"✅ {len(formatted)}개 문서 포맷 완료 (총 {len(result)}자)")
    return result

def create_chain_with_kb(retriever):
    """RAG 체인 생성 - Retriever로 문서 검색 후 LLM에 전달"""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """다음 문맥(context)을 참고하여 질문에 답변하세요.
문맥에 답이 없으면 모른다고 답하세요.

답변 시 반드시 어느 문서의 어느 페이지를 참고했는지 출처를 명시하세요.
예시: "[출처: manual.pdf 3페이지]"

Context:
{context}
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    def retrieve_and_format(x):
        """검색 실행 및 포맷팅"""
        try:
            input_text = x["input"] if isinstance(x, dict) else x
            print(f"\n🔍 검색 쿼리: '{input_text}'")

            retrieved_docs = retriever.invoke(input_text)
            print(f"📊 검색 결과: {len(retrieved_docs) if retrieved_docs else 0}개")

            return format_docs(retrieved_docs)
        except Exception as e:
            print(f"❌ 검색 오류: {e}")
            return f"검색 중 오류 발생: {str(e)}"

    # 체인 구성: 검색 → 프롬프트 → LLM
    return (
        {
            "context": retrieve_and_format,
            "chat_history": lambda x: x["chat_history"],
            "input": lambda x: x["input"],
        }
        | prompt
        | llm
    )


def create_chain_without_kb():
    """일반 대화용 체인 - KB 없이 LLM만 사용"""
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    return prompt | llm

def load_and_chunk_pdfs():
    """
    data 폴더의 모든 PDF를 로드하고 청크로 분할

    Returns:
        tuple: (chunks, status_message)
    """
    try:
        # data 폴더의 모든 PDF 파일 찾기
        pdf_files = list(Path(DATA_DIR).glob("*.pdf"))

        if not pdf_files:
            return [], "❌ data 폴더에 PDF 파일이 없습니다."

        all_chunks = []

        # 각 PDF 파일 처리
        for pdf_path in pdf_files:
            print(f"📄 로딩 중: {pdf_path.name}")

            # PDF 로드
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()

            # 텍스트 분할기 설정
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, add_start_index=True
            )
            
            # 문서를 청크로 분할
            chunks = text_splitter.split_documents(docs)

            # 메타데이터에 파일명 추가
            for chunk in chunks:
                chunk.metadata["source"] = pdf_path.name

            all_chunks.extend(chunks)
            print(f"✅ {pdf_path.name}: {len(chunks)}개 청크 생성")

        status_msg = f"✅ 성공!\n- PDF 파일 수: {len(pdf_files)}\n- 총 청크 수: {len(all_chunks)}"
        return all_chunks, status_msg

    except Exception as e:
        error_msg = f"❌ 오류 발생: {str(e)}"
        print(error_msg)
        return [], error_msg


def save_to_vectorstore(chunks, embeddings):
    """
    청크를 벡터스토어에 저장 (기존 데이터 삭제 후 저장)

    Args:
        chunks (list): 문서 청크 리스트
        embeddings: 임베딩 모델

    Returns:
        tuple: (vector_store, ids)
    """

    print("💾 벡터스토어에 저장 중...")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )

    # 기존 데이터 삭제 (파일은 유지)
    try:
        existing_ids = vector_store.get()["ids"]
        if existing_ids:
            print(f"🗑️  기존 데이터 {len(existing_ids)}개 삭제 중...")
            vector_store.delete(ids=existing_ids)
            print("✅ 기존 데이터 삭제 완료")
    except Exception as e:
        print(f"⚠️  기존 데이터 확인 중 오류 (무시하고 진행): {e}")

    # 문서를 벡터로 변환하여 저장
    ids = vector_store.add_documents(documents=chunks)

    print(f"✅ {len(ids)}개 문서를 벡터스토어에 저장 완료")
    return vector_store, ids

def reload_vectorstore():
    """
    벡터스토어 리로드 (PDF 로드 → 청킹 → 임베딩 → 저장)

    Returns:
        str: 실행 결과 메시지
    """
    global _retriever_cache

    try:
        # 1단계: PDF 로드 및 청킹
        chunks, status = load_and_chunk_pdfs()

        if not chunks:
            return status

        # 2단계: 임베딩 모델 가져오기 (캐시 사용)
        embeddings = get_embeddings()

        # 3단계: 벡터스토어에 저장
        vector_store, ids = save_to_vectorstore(chunks, embeddings)

        # 4단계: Retriever 캐시 무효화 (새로운 데이터 반영)
        _retriever_cache = None
        print("🔄 Retriever 캐시 초기화 (다음 검색 시 새 데이터 반영)")

        # 성공 메시지
        result = f"""✅ Knowledge Base 리로드 완료!

📊 처리 결과:
{status}
- 저장된 문서 ID 수: {len(ids)}
- 저장 위치: ./chroma_langchain_db
"""
        return result

    except Exception as e:
        error_msg = f"❌ 벡터스토어 저장 중 오류 발생:\n{str(e)}"
        print(error_msg)
        return error_msg

# 전역 변수: RAG 사용 여부
use_rag = False

def chatbot(message, history):
    """
    챗봇 응답 생성 (RAG 사용 여부에 따라 분기)

    Args:
        message (str): 사용자 입력
        history (list): 대화 히스토리

    Yields:
        str: 스트리밍 응답
    """
    # LangChain 메시지 형식으로 변환
    langchain_messages = []

    # 대화 히스토리를 LangChain 형식으로 변환
    for msg in history:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        else:
            langchain_messages.append(AIMessage(content=msg["content"]))

    # RAG 사용 여부에 따라 체인 선택
    if use_rag:
        retriever = get_retriever()
        if retriever is None:
            # 벡터DB가 비어있거나 초기화 실패
            error_msg = """⚠️ RAG 모드가 활성화되어 있지만 Knowledge Base를 사용할 수 없습니다.

다음 단계를 따라주세요:
1. data/ 폴더에 PDF 파일이 있는지 확인하세요
2. 우측 사이드바에서 'Knowledge Base Sync' 버튼을 클릭하세요
3. 리로드 완료 후 다시 질문해주세요"""
            yield error_msg
            return
        chain = create_chain_with_kb(retriever)
    else:
        chain = create_chain_without_kb()

    # 체인 입력 준비
    chain_input = {
        "chat_history": langchain_messages,
        "input": message,
    }

    # 스트리밍 응답 생성
    full_response = ""
    for chunk in chain.stream(chain_input):
        for char in chunk.content:
            full_response += char
            yield full_response + "▌"

    # 마지막: 커서 제거
    yield full_response

def toggle_rag(use_rag_mode):
    """
    RAG 모드 토글

    Args:
        use_rag_mode (bool): RAG 사용 여부
    """
    global use_rag
    use_rag = use_rag_mode
    print(f"🔄 RAG 모드: {'활성화' if use_rag else '비활성화'}")

with gr.Blocks() as demo:
    gr.Markdown("# 📚 LangChain을 활용한 RAG 기반 챗봇")

    with gr.Row():
        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=chatbot,
                type="messages"
            )

        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 설정")

            # RAG 모드 토글
            rag_toggle = gr.Checkbox(
                label="RAG 모드 사용",
                value=False,
                info=" Knowledge Base를 활용한 문서 기반 답변"
            )

            # RAG 토글 이벤트
            rag_toggle.change(
                fn=toggle_rag,
                inputs=rag_toggle
            )

            gr.Markdown("---")
            gr.Markdown("### 📦 Knowledge Base 관리")

            reload_btn = gr.Button("🔄 Knowledge Base Sync", variant="primary")
            status_output = gr.Textbox(
                label="상태",
                lines=5,
                interactive=False
            )

            # 버튼 클릭 이벤트
            reload_btn.click(
                fn=reload_vectorstore,
                outputs=status_output
            )

            gr.Markdown("""
            **사용 방법:**
            1. `data/` 폴더에 PDF 파일 추가
            2. 'Knowledge Base Sync' 버튼 클릭
            3. 'RAG 모드 사용' 체크
            4. 문서 기반 질문 시작
            """)

if __name__ == "__main__":
    demo.launch()