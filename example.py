import argparse
from pathlib import Path

from RAG.Embeddings import RuriV3Embedding
from RAG.LLM import DEFAULT_LOCAL_MODEL, LocalHFChat, OpenAIChat
from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles


DATA_DIR = "./data/phase5"
STORAGE_DIR = "./storage/phase5-ruri-v3-130m"
TOP_K = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval and optionally generate a grounded local answer."
    )
    parser.add_argument(
        "--question",
        help="Japanese question to answer from the Top-3 retrieved chunks.",
    )
    parser.add_argument(
        "--generator",
        choices=("local", "openai"),
        default="local",
        help="Answer generator to use. Defaults to local.",
    )
    parser.add_argument(
        "--local-model",
        default=DEFAULT_LOCAL_MODEL,
        help="Local Hugging Face causal language model name or directory.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens generated for the answer.",
    )
    return parser.parse_args()

TEST_CASES = [
    {
        "question": "都筑関耕地地区地区計画の目標は何ですか？",
        "expected_document": "都筑関耕地地区地区計画",
        "expected_district": None,
        "expected_item": "地区計画の目標",
        "expected_value": "国道246号線、都市計画道路日吉・元石川線",
    },
    {
        "question": "都筑関耕地地区のＡ－１地区では、どのような用途の建築物を建築できますか？",
        "expected_document": "都筑関耕地地区地区計画",
        "expected_district": "Ａ－１地区",
        "expected_item": "建築物の用途の制限",
        "expected_value": "学校、図書館その他これらに類するもの",
    },
    {
        "question": "都筑関耕地地区のＡ－１地区では、建築物の敷地面積の最低限度はいくらですか？",
        "expected_document": "都筑関耕地地区地区計画",
        "expected_district": "Ａ－１地区",
        "expected_item": "建築物の敷地面積の最低限度",
        "expected_value": "150㎡以上かつ住戸数に65㎡を乗じた面積以上",
    },
    {
        "question": "都筑関耕地地区のＢ－１地区では、建築物を前面道路から何メートル離す必要がありますか？",
        "expected_document": "都筑関耕地地区地区計画",
        "expected_district": "Ｂ－１地区",
        "expected_item": "壁面の位置の制限",
        "expected_value": "前面道路の境界線までの距離は2ｍ以上",
    },
    {
        "question": "港北ニュータウン中央地区地区計画の目標は何ですか？",
        "expected_document": "港北ニュータウン中央地区地区計画",
        "expected_district": None,
        "expected_item": "地区計画の目標",
        "expected_value": "市営地下鉄３号線センター南駅とセンター北駅の中間",
    },
    {
        "question": "港北ニュータウン中央地区の工場Ａ地区では、どのような用途の建築物が制限されていますか？",
        "expected_document": "港北ニュータウン中央地区地区計画",
        "expected_district": "工場Ａ地区",
        "expected_item": "建築物の用途の制限",
        "expected_value": "共同住宅，長屋，寄宿舎又は下宿",
    },
    {
        "question": "港北ニュータウン中央地区の工場Ｂ地区では、兼用住宅を建築できますか？",
        "expected_document": "港北ニュータウン中央地区地区計画",
        "expected_district": "工場Ｂ地区",
        "expected_item": "建築物の用途の制限",
        "expected_value": "住宅で事務所，店舗その他これらに類する用途を兼ねるもの",
    },
    {
        "question": "港北ニュータウン中央地区の沿道施設地区では、建築物の敷地面積の最低限度はいくらですか？",
        "expected_document": "港北ニュータウン中央地区地区計画",
        "expected_district": "沿道施設地区",
        "expected_item": "建築物の敷地面積の最低限度",
        "expected_value": "700㎡以上",
    },
    {
        "question": "鶴見潮田・本町通街並み誘導地区のＡ地区では、建築物の容積率の最高限度はいくらですか？",
        "expected_document": "鶴見潮田・本町通街並み誘導地区地区計画",
        "expected_district": "Ａ地区",
        "expected_item": "建築物の容積率の最高限度",
        "expected_value": "10分の20",
    },
    {
        "question": "鶴見潮田・本町通街並み誘導地区のＢ地区では、道路境界線から壁面をどれだけ離しますか？",
        "expected_document": "鶴見潮田・本町通街並み誘導地区地区計画",
        "expected_district": "Ｂ地区",
        "expected_item": "壁面の位置の制限",
        "expected_value": "距離は0.5ｍ以上",
    },
    {
        "question": "鶴見潮田・本町通街並み誘導地区のＣ地区では、建築物の高さの最高限度は何メートルですか？",
        "expected_document": "鶴見潮田・本町通街並み誘導地区地区計画",
        "expected_district": "Ｃ地区",
        "expected_item": "建築物の高さの最高限度",
        "expected_value": "31ｍを超えてはならない",
    },
    {
        "question": "日本大通り用途誘導地区地区計画の目標は何ですか？",
        "expected_document": "日本大通り用途誘導地区地区計画",
        "expected_district": None,
        "expected_item": "地区計画の目標",
        "expected_value": "賑わい形成の中心的地区「開港シンボル軸」",
    },
    {
        "question": "日本大通り用途誘導地区のＡ地区では、建築物の建ぺい率の最高限度はいくらですか？",
        "expected_document": "日本大通り用途誘導地区地区計画",
        "expected_district": "Ａ地区",
        "expected_item": "建築物の建ぺい率の最高限度",
        "expected_value": "10分の8",
    },
    {
        "question": "日本大通り用途誘導地区のＢ地区では、低層階を住居として使用できますか？",
        "expected_document": "日本大通り用途誘導地区地区計画",
        "expected_district": "Ｂ地区",
        "expected_item": "建築物の用途の制限",
        "expected_value": "2階以下の階を住居の用に供するもの",
    },
    {
        "question": "日本大通り用途誘導地区のＢ地区では、建築物の高さの最高限度は何メートルですか？",
        "expected_document": "日本大通り用途誘導地区地区計画",
        "expected_district": "Ｂ地区",
        "expected_item": "建築物の高さの最高限度",
        "expected_value": "75ｍ",
    },
    {
        "question": "綱島サスティナブル・スマートタウン地区のC地区では、どのような用途の建築物が制限されていますか？",
        "expected_document": "綱島サスティナブル・スマートタウン地区地区計画",
        "expected_district": "C地区",
        "expected_item": "建築物の用途の制限",
        "expected_value": "倉庫業を営むための倉庫",
    },
    {
        "question": "綱島サスティナブル・スマートタウン地区のC地区では、建築物の建ぺい率の最高限度はいくらですか？",
        "expected_document": "綱島サスティナブル・スマートタウン地区地区計画",
        "expected_district": "C地区",
        "expected_item": "建築物の建ぺい率の最高限度",
        "expected_value": "10分の5",
    },
    {
        "question": "綱島サスティナブル・スマートタウン地区のC地区では、建築物の敷地面積の最低限度はいくらですか？",
        "expected_document": "綱島サスティナブル・スマートタウン地区地区計画",
        "expected_district": "C地区",
        "expected_item": "建築物の敷地面積の最低限度",
        "expected_value": "300㎡",
    },
    {
        "question": "綱島サスティナブル・スマートタウン地区のC地区では、建築物を前面道路から何メートル離す必要がありますか？",
        "expected_document": "綱島サスティナブル・スマートタウン地区地区計画",
        "expected_district": "C地区",
        "expected_item": "壁面の位置の制限",
        "expected_value": "前面道路の境界線から5m以上",
    },
    {
        "question": "綱島サスティナブル・スマートタウン地区のC地区では、建築物の高さの最高限度は何メートルですか？",
        "expected_document": "綱島サスティナブル・スマートタウン地区地区計画",
        "expected_district": "C地区",
        "expected_item": "建築物の高さの最高限度",
        "expected_value": "建築物の高さは31m以下",
    },
]


args = parse_args()


def chunk_matches_expected(chunk, test_case):
    expected_fields = [
        f"資料名：{test_case['expected_document']}",
        f"項目：{test_case['expected_item']}",
        test_case["expected_value"],
    ]

    if test_case["expected_district"] is not None:
        expected_fields.append(f"地区：{test_case['expected_district']}")

    return all(expected_field in chunk for expected_field in expected_fields)


print("=== 1. BUILD INDEX ===")

reader = ReadFiles(DATA_DIR)
reader.file_list = sorted(reader.file_list)
if not reader.file_list:
    raise RuntimeError(f"No supported documents found in {DATA_DIR}.")

print(f"Documents found: {len(reader.file_list)}")
print("Source files:")
for source_path in reader.file_list:
    print(f"- {Path(source_path).name}")

chunks = reader.get_content(max_token_len=600, cover_content=150)
if not chunks:
    raise RuntimeError("No chunks were created from the regulation documents.")

print(f"Total chunks created: {len(chunks)}")

# Use this exact model instance for both indexing and querying.
embedding = RuriV3Embedding()
print(f"Embedding model: {embedding.path}")
print(f"Document prefix: {embedding.document_prefix}")
print(f"Query prefix: {embedding.query_prefix}")

index_store = VectorStore(chunks)
vectors = index_store.get_vector(EmbeddingModel=embedding)
index_store.persist(path=STORAGE_DIR)

print(f"Vectors created: {len(vectors)}")
print(f"Embedding dimension: {len(vectors[0]) if vectors else 0}")
print(f"Index saved to: {STORAGE_DIR}")


print("\n=== 2. RETRIEVAL TEST ===")

retrieval_store = VectorStore()
retrieval_store.load_vector(STORAGE_DIR)

evaluation_results = []

for test_number, test_case in enumerate(TEST_CASES, start=1):
    question = test_case["question"]

    retrieved_chunks = retrieval_store.query(
        question,
        EmbeddingModel=embedding,
        k=TOP_K,
    )

    print(f"\n=== TEST {test_number} ===")
    print(f"Question: {question}")
    print(f"Expected document: {test_case['expected_document']}")
    print(f"Expected district: {test_case['expected_district']}")
    print(f"Expected item: {test_case['expected_item']}")
    print(f"Expected value: {test_case['expected_value']}")

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        print(f"\n--- Rank {rank} ---")
        print(chunk)

    hit_at_1 = bool(retrieved_chunks) and chunk_matches_expected(
        retrieved_chunks[0], test_case
    )
    hit_at_3 = any(
        chunk_matches_expected(chunk, test_case)
        for chunk in retrieved_chunks
    )
    evaluation_results.append(
        {
            "test_case": test_case,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
        }
    )


print("\n=== BASELINE SUMMARY ===")

for test_number, result in enumerate(evaluation_results, start=1):
    test_case = result["test_case"]
    print(f"\nTest {test_number}")
    print(f"Question: {test_case['question']}")
    print(f"Expected document: {test_case['expected_document']}")
    print(f"Expected district: {test_case['expected_district']}")
    print(f"Expected item: {test_case['expected_item']}")
    print(f"Hit@1: {result['hit_at_1']} | Hit@3: {result['hit_at_3']}")

hit_at_1_count = sum(result["hit_at_1"] for result in evaluation_results)
hit_at_3_count = sum(result["hit_at_3"] for result in evaluation_results)
test_count = len(evaluation_results)

hit_at_1_accuracy = hit_at_1_count / test_count * 100
hit_at_3_accuracy = hit_at_3_count / test_count * 100

print(
    f"\nHit@1 Accuracy: {hit_at_1_count}/{test_count} "
    f"({hit_at_1_accuracy:.1f}%)"
)
print(
    f"Hit@3 Accuracy: {hit_at_3_count}/{test_count} "
    f"({hit_at_3_accuracy:.1f}%)"
)


if args.question:
    print("\n=== 3. TOP-3 ANSWER GENERATION ===")
    print(f"Question: {args.question}")

    answer_chunks = retrieval_store.query(
        args.question,
        EmbeddingModel=embedding,
        k=TOP_K,
    )

    print(f"Retrieved chunks: {len(answer_chunks)}")
    print(f"Generator: {args.generator}")

    if args.generator == "openai":
        answer_generator = OpenAIChat(model="gpt-5.6-luna")
        print(f"OpenAI model: {answer_generator.model}")
    else:
        print(f"Local LLM: {args.local_model}")
        print("Loading the local LLM. The first run may download model files.")
        answer_generator = LocalHFChat(
            path=args.local_model,
            max_new_tokens=args.max_new_tokens,
        )

    result = answer_generator.answer(args.question, answer_chunks)

    print("\nAnswer:")
    print(result.answer)
    print("\nCitations:")
    if result.citations:
        for citation in result.citations:
            print(citation.format())
    else:
        print("No supporting source was retrieved.")
