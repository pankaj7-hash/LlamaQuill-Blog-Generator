import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

st.set_page_config(
    page_title="🦙 LlamaQuill — by Pankaj Mahure",
    page_icon="🪶",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("🦙 LlamaQuill — Blog Generator")
st.caption("Generate thoughtful, audience-tuned blogs with local Ollama models — built by Pankaj Mahure")

# ---- UI ----
topic = st.text_input("✏️ Enter Blog Topic", placeholder="e.g., Top 5 AI Trends to Watch in 2025")
c1, c2 = st.columns(2)
with c1:
    words_str = st.text_input("🔢 Desired Word Count", value="500")
with c2:
    audience = st.selectbox("🎯 Target Audience", ("Researchers", "Data Scientists", "General Audience"), index=2)

with st.expander("⚙️ Advanced"):
    model_name = st.selectbox("Ollama model", ["llama3", "mistral", "phi3", "qwen2.5"], index=0)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)
    top_p = st.slider("top_p", 0.1, 1.0, 0.9, 0.05)
    num_predict = st.number_input("Max tokens to generate (num_predict)", 128, 4096, 700, step=50)
    base_url = st.text_input("Ollama base_url", value="http://localhost:11434")

generate = st.button("🚀 Generate Blog", type="primary")

# ---- Prompt ----
BLOG_PROMPT = PromptTemplate(
    input_variables=["audience", "topic", "words"],
    template=(
        "You are an expert content writer.\n"
        "Write a clear, engaging blog post for the audience: {audience}.\n"
        "Topic: {topic}\n"
        "Target length: about {words} words (±10%).\n\n"
        "Constraints:\n"
        "- Inviting intro, 2–4 short sections with headings, concise conclusion.\n"
        "- Short paragraphs (3–4 lines), concrete tips/examples, no fluff.\n"
        "- No code unless essential.\n"
        "Now write the blog."
    ),
)

def validate(topic: str, words_str: str):
    if not topic.strip():
        st.error("Please enter a blog topic.")
        return None
    try:
        w = int(words_str)
        if not (50 <= w <= 2000):
            st.error("Please choose a word count between 50 and 2000.")
            return None
        return w
    except ValueError:
        st.error("Word count must be an integer.")
        return None

# ---- Inference ----
if generate:
    words = validate(topic, words_str)
    if words is not None:
        try:
            llm = ChatOllama(
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                num_predict=num_predict,
                base_url=base_url,
            )
            prompt_text = BLOG_PROMPT.format(audience=audience, topic=topic.strip(), words=str(words))
            with st.spinner(f"Generating with {model_name} via Ollama…"):
                result = llm.invoke([HumanMessage(content=prompt_text)])
            st.subheader("📜 Generated Blog")
            st.markdown(result.content)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.info("Is Ollama running and is the model pulled? Try: `ollama pull llama3` then `ollama serve`.")
            st.info("If Ollama runs on another machine, set the correct base_url.")
            
st.markdown("---")
st.markdown(
    "**Developed with ❤️ by [Pankaj Mahure]**  \n"
    "🧠 Powered by *LangChain + Streamlit + Ollama (local models)*"
)
