import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew
import os

# Show title and description
st.title("🏢 BSI Lead Generation Tool")
st.write(
    "Enter a company's website URL to analyze their technology infrastructure and generate a summary. "
    "To use this app, you need to provide a Groq API key. "
)

# Ask user for their Groq API key
groq_api_key = st.text_input("Groq API Key", type="password")
if not groq_api_key:
    st.info("Please add your Groq API key to continue.", icon="🗝️")
else:
    # Set up LLM
    os.environ["OPENAI_API_KEY"] = "NA"
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=groq_api_key)

    # Let the user input a website URL
    website_url = st.text_input(
        "Enter the company's website URL",
        placeholder="https://example.com"
    )

    if website_url:
        # Web scraping function
        def scrape_website(url):
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text()
                return text[:4000]  # Limit to first 4000 characters for this example
            except Exception as e:
                return f"Error scraping website: {str(e)}"

        # Scrape the website
        scraped_content = scrape_website(website_url)

        # Create the summarizing agent
        company_info_summarizer = Agent(
            role="You are an AI agent specializing in analyzing company websites to identify key information about their technology infrastructure.",
            goal="Summarize the company's technology infrastructure based on the scraped website data.",
            backstory="You were trained to quickly extract and synthesize relevant information from company websites.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        # Create the summarization task
        summarize_company = Task(
            description=f"Summarize the company's technology infrastructure based on the scraped data: {scraped_content}",
            expected_output="""
            A detailed, structured text file containing:

            1. Company overview
            2. Explicit information about the technology stack of the company's product/service, NOT the website (if available)
            3. Inferred technology stack  of the company's product/service, NOT the website based on website analysis
            4. Explicit information about IT infrastructure (if available)
            5. Inferred IT infrastructure based on company size, industry, and website performance
            6. Explicit information about hardware setup (if available)
            7. Inferred hardware setup based on company operations and industry standards
            8. Cloud usage indicators
            9. Potential scalability and performance issues
            10. Security measures and compliance indicators
            11. Integration with third-party services or APIs
            12. Any unique technological challenges or requirements based on the company's specific industry or operations

            This comprehensive report will provide a solid foundation for the next agent to identify potential hardware-related pain points and opportunities for technological optimization.
            """,
            agent=company_info_summarizer
        )

        # Create and run the crew
        crew = Crew(
            agents=[company_info_summarizer],
            tasks=[summarize_company],
            verbose=2
        )

        # Run the analysis when the user clicks the button
        if st.button("Analyze Website"):
            with st.spinner("Analyzing website..."):
                try:
                    result = crew.kickoff()
                    st.success("Analysis complete!")
                    st.write(result)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")