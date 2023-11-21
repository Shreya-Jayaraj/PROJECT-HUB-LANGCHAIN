import streamlit as st
import langchain_llms as lch
import textwrap

#st.title("Project Hub")
st.set_page_config(page_title="Project-Hub", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center; color: green;'>Project-Hub</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Perform queries on your PDF/Youtube Video</h1>", unsafe_allow_html=True)


# document_type = st.radio("Select the type of your document", ("Youtube Video", "PDF"))

# if document_type == "Youtube Video":
#     doc = st.text_area(
#         label="Paste the link",
#         max_chars=50
#     )
# else:
#     doc = st.file_uploader("Upload your PDF", type='pdf')
#     #if doc:
#     #   doc = doc.name

# query = st.text_area(
#     label="What is your query? ",
#     max_chars=50,
#     key="query"
# )
# openai_api_key = st.text_area(
#     label="OpenAI API Key",
#     key="langchain_search_api_key_openai",
#     max_chars=50,
#     type="password"
# )

document_type = st.radio("Select the type of your document", ("Youtube Video", "PDF"))

if document_type == "Youtube Video":
    doc = st.text_area(
        label="Paste the YouTube Video link",
        max_chars=100
    )
else:
    doc = st.file_uploader("Upload your PDF", type='pdf')

query = st.text_area(
    label="What is your query? ",
    max_chars=100,
    key="query"
)

openai_api_key = st.text_input(
    label="Enter your OpenAI API Key",
    key="langchain_search_api_key_openai",
    max_chars=50,
    type="password"
)

        
submit_button = st.button('Submit')
if submit_button:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    if query and doc and document_type == "Youtube Video":
        db = lch.create_vector_db_from_youtube_url(doc)
        response, docs = lch.get_response_from_query_for_youtube(db, query)
        st.subheader("Answer:")
        st.write(textwrap.fill(response, width=85))

    elif query and doc and document_type == "PDF":
        db = lch.create_vector_db_from_pdf(doc)
        response, docs = lch.get_response_from_query_for_pdf(db, query)
        st.subheader("Answer: ")
        st.write(textwrap.fill(response, width=85))