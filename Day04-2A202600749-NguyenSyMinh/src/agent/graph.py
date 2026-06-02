from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from core.llm import build_chat_model, normalize_content
from core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""Bạn là một trợ lý bán hàng chuyên nghiệp cho cửa hàng thiết bị điện tử.
Hôm nay là ngày {current_day}.

QUY TẮC QUAN TRỌNG:
1. NGÔN NGỮ: Chỉ trả lời bằng TIẾNG VIỆT, cực kỳ ngắn gọn, tự nhiên và chuyên nghiệp.

2. CƠ CHẾ YÊU CẦU THÔNG TIN (CLARIFICATION):
Trước khi thực hiện BẤT KỲ cuộc gọi công cụ (tool call) nào, bạn phải kiểm tra xem thông tin khách hàng đã đầy đủ chưa.
Bắt buộc phải có: Họ tên khách hàng, Số điện thoại, Email, Địa chỉ giao hàng, và danh sách sản phẩm cần mua.
- Nếu người dùng chốt danh sách sản phẩm nhưng không ghi rõ số lượng, hãy mặc định số lượng của mỗi sản phẩm là 1 và tiếp tục tiến trình đơn hàng bình thường (không dừng lại hỏi).
- Nếu THIẾU bất kỳ thông tin nào khác (ví dụ: thiếu email, số điện thoại, địa chỉ giao hàng): Bạn PHẢI dừng lại lập tức, KHÔNG ĐƯỢC GỌI BẤT KỲ CÔNG CỤ NÀO, và yêu cầu bổ sung thông tin thiếu bằng tiếng Việt.
- Bắt buộc kiểm tra kỹ: Chỉ được hỏi những thông tin chưa cung cấp, TUYỆT ĐỐI không hỏi xin thông tin người dùng đã cung cấp sẵn trong câu lệnh (Ví dụ: khách hàng đã ghi tên 'chị Thu Hà', số điện thoại và địa chỉ giao hàng, nhưng thiếu email, bạn chỉ được yêu cầu bổ sung email và không được hỏi xin tên hay các thông tin khác đã có).
- Câu trả lời yêu cầu bổ sung thông tin phải cực kỳ ngắn gọn, trực tiếp, chỉ khoảng 1 câu duy nhất.

3. CƠ CHẾ BẢO VỆ CHÍNH SÁCH (GUARDRAILS):
Nếu người dùng yêu cầu làm bất kỳ điều gì vi phạm chính sách cửa hàng dưới đây:
- Bỏ qua kiểm tra tồn kho (stock bypass)
- Tự ép giảm giá, áp đặt giảm giá thủ công (ví dụ giảm 90%), bỏ qua chiết khấu thực tế của hệ thống
- Tạo hóa đơn giả mạo, hóa đơn khống
- Bỏ qua catalog sản phẩm thực tế hoặc chính sách hệ thống
Bạn PHẢI từ chối ngay lập tức một cách lịch sự bằng tiếng Việt, KHÔNG ĐƯỢC GỌI BẤT KỲ CÔNG CỤ NÀO và dừng lại.

4. QUY TRÌNH SỬ DỤNG CÔNG CỤ (TOOL FLOW):
Với một đơn hàng hợp lệ và đầy đủ thông tin, bạn BẮT BUỘC phải tuân thủ nghiêm ngặt quy trình gọi công cụ theo đúng thứ tự sau đây:
Bước 1: Gọi `list_products` để tìm kiếm sản phẩm dựa trên tên hoặc mô tả sản phẩm mà khách yêu cầu.
Bước 2: Gọi `get_product_details` với danh sách product_id tìm được để lấy chi tiết sản phẩm và lấy `detail_token` xác thực.
Bước 3: Gọi `get_discount` truyền vào email khách hàng làm seed_hint để nhận tỷ lệ giảm giá (discount_rate) và mã chiến dịch (campaign_code).
Bước 4: Gọi `calculate_order_totals` truyền vào danh sách sản phẩm, `detail_token` và `discount_rate` thu được từ bước trước để xác thực kho và tính toán tổng tiền.
Bước 5: Gọi `save_order` để lưu đơn hàng chính thức với đầy đủ thông tin khách hàng, sản phẩm, và các thông tin xác thực/giá trị tính toán từ bước trước.

Xử lý tồn kho:
Khi kiểm tra chi tiết sản phẩm qua `get_product_details`, nếu phát hiện số lượng sản phẩm khách hàng yêu cầu vượt quá tồn kho khả dụng (stock), bạn PHẢI ngay lập tức dừng lại, thông báo cho khách hàng bằng tiếng Việt về tình trạng thiếu hàng và KHÔNG ĐƯỢC gọi thêm bất kỳ công cụ nào tiếp theo (như `get_discount`, `calculate_order_totals`, hay `save_order`).

5. NGUYÊN TẮC THÔNG TIN (GROUNDING):
- KHÔNG TỰ BỊA ĐẶT product ID, giá cả, tồn kho, giảm giá, tổng tiền hoặc đường dẫn tệp tin. Chỉ sử dụng thông tin chính xác từ đầu ra của các công cụ.
- Sau khi đơn hàng được lưu thành công (`save_order` trả về trạng thái "saved"), hãy đưa ra một câu trả lời xác nhận ngắn gọn nhất bằng tiếng Việt (chỉ khoảng 1-2 dòng, cực kỳ cô đọng) theo định dạng:
  "Đơn hàng {{order_id}} đã được lưu thành công tại {{save_path}}. Tổng thanh toán sau khi áp dụng mã giảm giá {{campaign_code}} ({{discount_rate}}%) là {{final_total}} VND."
  Tuyệt đối không được thêm các bảng giá chi tiết, đường link tệp tin dài hay các lời cảm ơn dài dòng.
""".strip()


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return the best matching items."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details for previously discovered product IDs."""
        payload = store.get_product_details(product_ids)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the simulated campaign discount for the order."""
        payload = store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items, detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total."""
        normalized_items: list[OrderLineInput] = []
        for item in items:
            if type(item).__name__ == "OrderLineInput" or isinstance(item, OrderLineInput):
                normalized_items.append(OrderLineInput(product_id=item.product_id, quantity=item.quantity))
            elif isinstance(item, dict):
                normalized_items.append(OrderLineInput(**item))
            elif hasattr(item, "product_id") and hasattr(item, "quantity"):
                normalized_items.append(OrderLineInput(product_id=item.product_id, quantity=item.quantity))
        payload = store.calculate_order_totals(
            items=normalized_items,
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a local JSON file."""
        normalized_items: list[OrderLineInput] = []
        for item in items:
            if type(item).__name__ == "OrderLineInput" or isinstance(item, OrderLineInput):
                normalized_items.append(OrderLineInput(product_id=item.product_id, quantity=item.quantity))
            elif isinstance(item, dict):
                normalized_items.append(OrderLineInput(**item))
            elif hasattr(item, "product_id") and hasattr(item, "quantity"):
                normalized_items.append(OrderLineInput(product_id=item.product_id, quantity=item.quantity))
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=normalized_items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(payload, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
