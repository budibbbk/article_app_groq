# --- Patch must come before phi import ---
import inspect
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
# -----------------------------------------

import nest_asyncio
from typing import Optional
from groq import Groq
import streamlit as st
from duckduckgo_search import DDGS   # ✅ correct class
from phi.tools.newspaper4k import Newspaper4k
import os
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Allow asyncio in Streamlit
nest_asyncio.apply()

# Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Streamlit config
st.set_page_config(
    page_title="News Article",
    page_icon=":orange_heart:",
)

st.title("📰 News Articles")
st.markdown("##### :orange_heart: built using Groq API")


def truncate_text(text: str, words: int) -> str:
    """Truncate text to the first N words"""
    return " ".join(text.split()[:words])


def main() -> None:

    # Sidebar model options
    summary_model = st.sidebar.selectbox(
        "Select summary model", options=["llama3-8b-8192", "llama3-70b-8192"]
    )

    writer_model = st.sidebar.selectbox(
        "Select writer model", options=["llama3-8b-8192", "llama3-70b-8192"]
    )

    st.sidebar.markdown("## Research Options")

    num_search_results = st.sidebar.slider(
        ":sparkles: Number of Search Results",
        min_value=2,
        max_value=10,
        value=5,
        help="Number of results to search for, note only the articles that can be read will be summarized",
    )

    per_article_summary = st.sidebar.slider(
        ":sparkles: Words per article summary",
        min_value=100,
        max_value=1500,
        value=800,
        step=100,
        help="Number of words per article summary",
    )

    new_summary_length = st.sidebar.slider(
        ":sparkles: Final summary length",
        min_value=1000,
        max_value=5000,
        value=4000,
        step=1000,
        help="Number of words for the final article",
    )

    article_topic = st.text_input("Search Query", key="search_query")
    Write_button = st.button("Write Article")

    if Write_button:
        news_results = []
        new_summary: Optional[str] = None

        # --- Step 1: Fetch News ---
        with st.status("🔍 Reading news...", expanded=True) as status:
            with st.container():
                news_container = st.empty()
                ddgs = DDGS()
                newspaper_tools = Newspaper4k()
                results = ddgs.news(keywords=article_topic)

                for r in results:
                    if "url" in r:
                        article_data = newspaper_tools.get_article_data(r["url"])

                        if article_data and "text" in article_data:
                            r["text"] = article_data["text"]
                            news_results.append(r)

                            # Live update in Streamlit
                            news_container.write(news_results)

            if news_results:
                news_container.write(news_results)
                status.update(label="✅ News search complete", state="complete", expanded=False)

        # --- Step 2: Summarize News ---
        if len(news_results) > 0:
            news_summary = ""
            with st.status("📝 Summarizing news...", expanded=True) as status:
                with st.container():
                    summary_container = st.empty()

                    for news_result in news_results:
                        news_summary += f"### {news_result.get('title','No title')}\n\n"
                        news_summary += f"- Date: {news_result.get('date','N/A')}\n\n"
                        news_summary += f"- URL: {news_result.get('url','')}\n\n"
                        news_summary += f"### Introduction\n\n{news_result.get('text','')[:500]}...\n\n"

                        # Groq API call
                        completion = groq_client.chat.completions.create(
                            model=summary_model,
                            messages=[
                                {"role": "system", "content": "Summarize the article clearly and concisely."},
                                {"role": "user", "content": news_result["text"]},
                            ],
                            max_tokens=per_article_summary,
                            temperature=0.6,
                        )

                        _summary = completion.choices[0].message.content.strip()

                        # Truncate if too long
                        _summary_length = len(_summary.split())
                        if _summary_length > new_summary_length:
                            _summary = truncate_text(_summary, new_summary_length)

                        news_summary += "### Summary\n\n"
                        news_summary += _summary
                        news_summary += "\n\n---\n\n"

                        # Live update
                        if news_summary:
                            summary_container.markdown(news_summary)

                        # Stop if final summary too long
                        if len(news_summary.split()) > new_summary_length:
                            break

                if news_summary:
                    summary_container.markdown(news_summary)
                status.update(label="✅ News summarization complete", state="complete", expanded=False)

            if news_summary is None:
                st.error("❌ No summary generated. Please try again.")
                return

        # --- Step 3: Generate Final Article Draft ---
        if news_summary:
            with st.status("📰 Generating final article draft...", expanded=True) as status:
                completion = groq_client.chat.completions.create(
                    model=writer_model,
                    messages=[
                        {"role": "system", "content": "You are a professional journalist. Write a well-structured, engaging article based on the provided summaries."},
                        {"role": "user", "content": f"Here are the summaries:\n\n{news_summary}\n\nWrite a detailed article draft with introduction, body, and conclusion."}
                    ],
                    max_tokens=new_summary_length,
                    temperature=0.7,
                )

                article_draft = completion.choices[0].message.content.strip()
                st.subheader("📝 Final Drafted Article")
                st.markdown(article_draft)

                status.update(label="✅ Article draft complete", state="complete", expanded=False)

                with st.spinner("writing article..."):
                    final_report = ""
                    final_report_container = st.empty()

                    response = groq_client.chat.completions.create(
                        model=writer_model,
                        messages=[
                            {"role": "system", "content": "Write the article."}
                        ],
                        max_tokens=new_summary_length,
                        temperature=0.7,
                    )

                    final_report = response.choices[0].message.content.strip()
                    final_report_container.markdown(final_report)

    st.sidebar.markdown("---")
    if st.sidebar.button("Restart"):
        st.rerun()


main()
