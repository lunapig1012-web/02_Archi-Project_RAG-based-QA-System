"""Mock responses for the Streamlit UI prototype.

The dictionaries returned here intentionally mirror the shape expected from a
future RAG backend. No embedding model, vector store, LLM, or API is used.
"""

from copy import deepcopy


SOURCE = {
    "title": "綱島サスティナブル・スマートタウン地区地区計画",
    "plan_id": "C-102",
    "url": (
        "https://www.city.yokohama.lg.jp/kurashi/machizukuri-kankyo/"
        "toshiseibi/plan-rule/chikukeikaku/kubetsu/kohoku/c-102.html"
    ),
}


MOCK_RESPONSES = {
    "height": {
        "answer": "C地区の建築物の高さは31m以下です。",
        "regulation": "建築物の高さの最高限度",
        "conditions": "C地区",
        "notes": (
            "建築物の各部分には、前面道路中心線までの真北方向の水平距離に"
            "基づく追加の高さ制限があります。"
        ),
        "source": SOURCE,
        "retrieved_chunks": [
            {
                "document": SOURCE["title"],
                "district": "C地区",
                "section": "建築物の高さの最高限度",
                "text": (
                    "1. 建築物の高さは31m以下とする。\n"
                    "2. 建築物の各部分の高さは、その部分から前面道路中心線までの"
                    "真北方向の水平距離に0.6を乗じ、その値に10mを加えた高さ以下とする。"
                ),
            },
            {
                "document": SOURCE["title"],
                "district": "C地区",
                "section": "地区の名称・面積",
                "text": "地区の名称：C地区\n面積：約0.4ha",
            },
            {
                "document": SOURCE["title"],
                "district": "",
                "section": "地区計画の目標",
                "text": "良好な居住機能や生活支援機能を適切に導入し、快適な市街地形成を図る。",
            },
        ],
    },
    "coverage": {
        "answer": "C地区の建築物の建ぺい率の最高限度は10分の5（50%）です。",
        "regulation": "建築物の建ぺい率の最高限度",
        "conditions": "C地区",
        "notes": "地区計画では、建ぺい率の最高限度が「10分の5」と定められています。",
        "source": SOURCE,
        "retrieved_chunks": [
            {
                "document": SOURCE["title"],
                "district": "C地区",
                "section": "建築物の建ぺい率の最高限度",
                "text": "10分の5",
            },
            {
                "document": SOURCE["title"],
                "district": "C地区",
                "section": "建築物の敷地面積の最低限度",
                "text": "300㎡",
            },
            {
                "document": SOURCE["title"],
                "district": "C地区",
                "section": "地区の名称・面積",
                "text": "地区の名称：C地区\n面積：約0.4ha",
            },
        ],
    },
    "setback": {
        "answer": "C地区では、建築物を前面道路の境界線から5m以上後退させる必要があります。",
        "regulation": "壁面の位置の制限",
        "conditions": "C地区",
        "notes": (
            "公衆便所、巡査派出所など公益上必要とされる建築物、またはその一部には、"
            "この制限が適用されない場合があります。"
        ),
        "source": SOURCE,
        "retrieved_chunks": [
            {
                "document": SOURCE["title"],
                "district": "C地区",
                "section": "壁面の位置の制限",
                "text": (
                    "建築物の外壁またはこれに代わる柱の面は、"
                    "前面道路の境界線から5m以上離す必要がある。"
                ),
            },
            {
                "document": SOURCE["title"],
                "district": "C地区",
                "section": "壁面の位置の制限",
                "text": (
                    "公衆便所、巡査派出所など公益上必要とされる建築物、"
                    "またはその一部は適用対象外となる。"
                ),
            },
            {
                "document": SOURCE["title"],
                "district": "C地区",
                "section": "地区の名称・面積",
                "text": "地区の名称：C地区\n面積：約0.4ha",
            },
        ],
    },
}


FALLBACK_RESPONSE = {
    "answer": "このUIプロトタイプでは、綱島C地区の代表的な規制をモック表示しています。",
    "regulation": "UIプロトタイプ（検索未接続）",
    "conditions": "高さ・建ぺい率・壁面後退の質問に対応",
    "notes": "実際の文書検索と回答生成は、今後のフェーズで接続します。",
    "source": SOURCE,
    "retrieved_chunks": [],
}


def get_mock_response(question: str) -> dict:
    """Return a backend-shaped mock result selected by question keywords."""

    normalized_question = question.strip()

    if "建ぺい率" in normalized_question or "建蔽率" in normalized_question:
        response = MOCK_RESPONSES["coverage"]
    elif any(keyword in normalized_question for keyword in ("後退", "道路", "壁面")):
        response = MOCK_RESPONSES["setback"]
    elif "高さ" in normalized_question:
        response = MOCK_RESPONSES["height"]
    else:
        response = FALLBACK_RESPONSE

    return deepcopy(response)
