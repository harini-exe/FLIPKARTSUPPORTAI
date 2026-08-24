import re
from typing import TypedDict, Optional

import chromadb
import joblib
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import models, transforms
from langgraph.graph import StateGraph, END


# ============================================================
# PART 3 — POLICY KNOWLEDGE BASE
# ============================================================

POLICY_DOCS = []

for path in sorted(__import__("pathlib").Path("knowledge_base").glob("doc_*.txt")):
    text = path.read_text().strip()
    doc_id = path.stem

    POLICY_DOCS.append({
        "id": doc_id,
        "title": doc_id,
        "text": text
    })


def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s.strip() for s in sentences if s.strip()]


chunks = []

for doc in POLICY_DOCS:
    sentences = split_into_sentences(doc["text"])

    for i, sentence in enumerate(sentences):
        chunks.append({
            "chunk_id": f"{doc['id']}_chunk{i}",
            "doc_id": doc["id"],
            "doc_title": doc["title"],
            "text": sentence
        })


# ============================================================
# VECTOR INDEX
# ============================================================

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.get_or_create_collection(
    name="flipkart_policies"
)

if collection.count() == 0:
    embeddings = embedder.encode(
        [c["text"] for c in chunks]
    )

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "doc_id": c["doc_id"],
                "doc_title": c["doc_title"]
            }
            for c in chunks
        ]
    )


def search(query, top_k=3):
    query_embedding = embedder.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )

    retrieved = []

    for i in range(len(results["documents"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "doc_id": results["metadatas"][0][i]["doc_id"],
            "doc_title": results["metadatas"][0][i]["doc_title"],
            "score": results["distances"][0][i]
        })

    return retrieved


# ============================================================
# TOOL 1 — RETURN RISK
# ============================================================

return_risk_pipeline = joblib.load(
    "models/return_risk_model.pkl"
)

T_STAR_RF = 0.46


def check_return_risk(order_features: dict) -> dict:
    order_df = pd.DataFrame([order_features])

    prob = return_risk_pipeline.predict_proba(
        order_df
    )[0, 1]

    if prob < T_STAR_RF:
        bucket = "Low"
    elif prob < T_STAR_RF + 0.15:
        bucket = "Medium"
    else:
        bucket = "High"

    return {
        "return_probability": round(float(prob), 4),
        "risk_bucket": bucket
    }


# ============================================================
# TOOL 2 — PRODUCT IMAGE CLASSIFIER
# ============================================================

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def load_classifier(
    weights_path="models/product_classifier.pt",
    device="cpu"
):
    backbone = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    backbone.fc = nn.Identity()
    backbone.eval()
    backbone.to(device)

    head = nn.Linear(512, 10)

    head.load_state_dict(
        torch.load(
            weights_path,
            map_location=device
        )
    )

    head.eval()
    head.to(device)

    return backbone, head


def predict_image(
    image_path,
    backbone,
    head,
    device="cpu"
):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open(image_path).convert("L")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = backbone(img_tensor)
        outputs = head(features)

        probs = torch.softmax(outputs, dim=1)

        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    return {
        "predicted_category": class_names[pred_idx],
        "confidence": confidence
    }


_backbone, _head = load_classifier()


def classify_product_image(image_path: str) -> dict:
    result = predict_image(
        image_path,
        _backbone,
        _head
    )

    return {
        "predicted_category": result["predicted_category"],
        "confidence": round(
            result["confidence"],
            4
        )
    }


# ============================================================
# LANGGRAPH AGENT
# ============================================================

class AgentState(TypedDict):
    user_input: str
    intent: Optional[str]
    retrieved_chunks: Optional[list]
    tool_output: Optional[dict]
    last_order_id: Optional[str]
    final_answer: Optional[dict]


SIMILARITY_THRESHOLD = 1.0


INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all rules",
    "ignore the above",
    "pretend you are",
    "disregard your instructions",
    "act as if you have no rules",
    "forget your instructions"
]


def check_prompt_injection(user_input: str) -> bool:
    text = user_input.lower()

    return any(
        pattern in text
        for pattern in INJECTION_PATTERNS
    )


def intent_node(state: AgentState) -> AgentState:

    text = state["user_input"].lower()

    if check_prompt_injection(
        state["user_input"]
    ):
        state["intent"] = "blocked"
        return state

    if (
        "image" in text
        or "photo" in text
        or ".png" in text
        or "picture" in text
    ):
        intent = "product_category"

    elif (
        "risk" in text
        or "returned" in text
        or (
            "order" in text
            and (
                "likely" in text
                or "predict" in text
            )
        )
        or (
            "order" in text
            and state.get("last_order_id")
        )
    ):
        intent = "return_risk"

    else:
        intent = "policy"

    state["intent"] = intent

    return state


def rag_retrieval_node(state: AgentState) -> AgentState:

    state["retrieved_chunks"] = search(
        state["user_input"],
        top_k=3
    )

    return state


def tool_calling_node(state: AgentState) -> AgentState:

    if state["intent"] == "return_risk":

        sample_order = {
            "product_category": "Footwear",
            "price_inr": 2200,
            "discount_pct": 35,
            "payment_method": "COD",
            "customer_tenure_days": 60,
            "num_previous_orders": 2,
            "num_previous_returns": 1,
            "delivery_distance_km": 300,
            "delivery_days": 5,
            "is_weekend_order": 0,
            "rating_given": 3.0
        }

        state["tool_output"] = check_return_risk(
            sample_order
        )

        state["last_order_id"] = "order_4521"

    elif state["intent"] == "product_category":

        state["tool_output"] = classify_product_image(
            "data/sample_images/04_Shirt.png"
        )

    return state


def response_generation_node(
    state: AgentState
) -> AgentState:

    intent = state["intent"]

    if intent == "blocked":

        state["final_answer"] = {
            "answer":
                "I can't follow instructions embedded "
                "in your message that try to override my "
                "guidelines. Please ask your support "
                "question normally.",
            "source": "policy_kb",
            "confidence": 0.0
        }

        return state

    if intent == "policy":

        chunks_found = state["retrieved_chunks"]

        if not chunks_found:
            state["final_answer"] = {
                "answer":
                    "I'm not confident I have a grounded "
                    "policy answer for that.",
                "source": "policy_kb",
                "confidence": 0.0
            }

            return state

        top_score = chunks_found[0]["score"]

        if top_score > SIMILARITY_THRESHOLD:

            state["final_answer"] = {
                "answer":
                    "I'm not confident I have a grounded "
                    "policy answer for that. Could you "
                    "rephrase, or ask a support agent "
                    "directly?",
                "source": "policy_kb",
                "confidence": 0.0
            }

        else:

            best_chunk = chunks_found[0]

            state["final_answer"] = {
                "answer": best_chunk["text"],
                "source": "policy_kb",
                "confidence":
                    round(
                        1 - min(
                            top_score,
                            1.0
                        ),
                        4
                    )
            }

    elif intent == "return_risk":

        tool_result = state["tool_output"]

        state["final_answer"] = {
            "answer":
                f"This order has a "
                f"{tool_result['return_probability'] * 100:.1f}% "
                f"predicted return probability, which falls "
                f"in the '{tool_result['risk_bucket']}' "
                f"risk bucket.",
            "source": "return_risk_tool",
            "confidence":
                tool_result["return_probability"]
        }

    elif intent == "product_category":

        tool_result = state["tool_output"]

        state["final_answer"] = {
            "answer":
                f"This image is predicted to be a "
                f"'{tool_result['predicted_category']}' "
                f"with "
                f"{tool_result['confidence'] * 100:.1f}% "
                f"confidence.",
            "source": "image_classifier_tool",
            "confidence":
                tool_result["confidence"]
        }

    return state


def route_after_intent(state: AgentState) -> str:

    if state["intent"] == "policy":
        return "rag"

    elif state["intent"] in (
        "return_risk",
        "product_category"
    ):
        return "tool"

    else:
        return "respond"


graph = StateGraph(AgentState)

graph.add_node(
    "intent",
    intent_node
)

graph.add_node(
    "rag",
    rag_retrieval_node
)

graph.add_node(
    "tool",
    tool_calling_node
)

graph.add_node(
    "respond",
    response_generation_node
)

graph.set_entry_point("intent")

graph.add_conditional_edges(
    "intent",
    route_after_intent,
    {
        "rag": "rag",
        "tool": "tool",
        "respond": "respond"
    }
)

graph.add_edge(
    "rag",
    "respond"
)

graph.add_edge(
    "tool",
    "respond"
)

graph.add_edge(
    "respond",
    END
)

app = graph.compile()


def run_agent(
    user_input,
    prior_state=None
):

    state = prior_state or {
        "user_input": "",
        "intent": None,
        "retrieved_chunks": None,
        "tool_output": None,
        "last_order_id": None,
        "final_answer": None
    }

    state["user_input"] = user_input

    return app.invoke(state)


if __name__ == "__main__":

    print(
        run_agent(
            "How long do I have to return shoes?"
        )["final_answer"]
    )
