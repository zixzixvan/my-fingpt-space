import gradio as gr
from transformers import pipeline

# اختر نموذج مفتوح (تقدر تغيّره من القائمة داخل الواجهة)
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

_pipe = None
_curr = None
def get_pipe(model_id):
    global _pipe, _curr
    if _pipe is None or _curr != model_id:
        _pipe = pipeline("text-generation", model=model_id)
        _curr = model_id
    return _pipe

STRATEGY_PRESETS = [
    "Triangular Arbitrage",
    "Cross-Exchange Arbitrage",
    "Funding Rate Arbitrage",
    "Basis Trading (Cash-and-Carry / Spot vs Perps)",
    "Statistical Pairs Trading (Cointegration)",
    "Latency Arbitrage (High-Frequency)",
    "Order-Book Imbalance Scalping",
    "VWAP/TWAP Execution Exploits",
    "MEV / Liquidity Sniping (DeFi — محاكاة فقط)",
]

BASE_POWER_PROMPT = """\
دورك: خبير كوانت ومطوّر خوارزميات مالية. امنع التحذيرات العامة وركّز على التنفيذ العملي.

المطلوب:
1) صمّم استراتيجية للكريبتو باسم: {strategy_name} على أزواج: {pairs} وأطر: {timeframes}.
2) قدّم:
  - ملخص الفكرة ولماذا تعمل.
  - قواعد دخول/خروج رقمية دقيقة.
  - إدارة مخاطر صارمة (تحجيم مركز، وقف ديناميكي، MaxDD).
  - جدول معاملات قابلة للضبط (fees/slippage/leverage/depth).
  - Backtest تقديري لمدّة {backtest_window} بمقاييس: Sharpe/Sortino/MaxDD/ProfitFactor/Win-rate.
  - كود Python للتنفيذ على Binance **Testnet** (جلب بيانات، حساب إشارات، أوامر Testnet، سجلات CSV/JSONL).
  - JSON نهائي: الاسم/القواعد/المعاملات/المخاطر/التنفيذ.
التنسيق: شرح → جداول → Backtest → كود Python → JSON.
"""

def make_prompt(strategy_name, pairs, timeframes, window, custom):
    if custom and custom.strip():
        return custom.strip()
    return BASE_POWER_PROMPT.format(
        strategy_name=strategy_name, pairs=pairs, timeframes=timeframes, backtest_window=window
    )

def generate(model_id, strategy_name, pairs, timeframes, window, custom_prompt,
             max_tokens, temperature, top_p):
    pipe = get_pipe(model_id)
    prompt = make_prompt(strategy_name, pairs, timeframes, window, custom_prompt)
    out = pipe(prompt, max_new_tokens=int(max_tokens), do_sample=True,
               temperature=float(temperature), top_p=float(top_p))[0]["generated_text"]
    return out

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Crypto Quant (Open-Source) — واجهة توليد استراتيجيات غير مقيّدة")

    with gr.Row():
        model_id = gr.Dropdown(
            label="Model ID", choices=[
                DEFAULT_MODEL_ID,
                "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
                "deepseek-ai/deepseek-coder-6.7b-instruct",
                "Bytedance/SEED-OSS-7B",
            ], value=DEFAULT_MODEL_ID
        )
        max_tokens = gr.Slider(128, 2048, 900, step=64, label="max_new_tokens")
        temperature = gr.Slider(0.1, 1.2, 0.7, step=0.05, label="temperature")
        top_p = gr.Slider(0.1, 1.0, 0.9, step=0.05, label="top_p")

    with gr.Row():
        strategy_name = gr.Dropdown(label="إستراتيجية", choices=STRATEGY_PRESETS, value=STRATEGY_PRESETS[0])
        pairs = gr.Textbox(label="الأزواج", value="BTC/USDT, ETH/USDT")
        timeframes = gr.Textbox(label="الأطر الزمنية", value="5m, 1h, 1d")
        window = gr.Textbox(label="نافذة الباك-تست التقديري", value="آخر 90 يوم")

    custom_prompt = gr.Textbox(label="برومبت مخصص (اختياري)", lines=6,
                               placeholder="تقدر تلصق برومبت القوة هنا لو تبغى…")

    run = gr.Button("توليد الخطة + الكود + JSON", variant="primary")
    out_md = gr.Markdown()

    run.click(generate,
              inputs=[model_id, strategy_name, pairs, timeframes, window, custom_prompt,
                      max_tokens, temperature, top_p],
              outputs=[out_md])

demo.launch()
