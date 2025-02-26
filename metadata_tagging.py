from langchain_ollama import OllamaLLM, ChatOllama
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
import logging

logging.basicConfig(level=logging.INFO, filename="metadatalog.log", filemode="w", format="%(levelname)s: %(message)s")

model = ChatOllama(model="qwen2.5:32b", temperature=0.0)

prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""You are an expert in fintech documentation analysis. 
    Below is a text snippet from our fintech company's documentation (which could be an FAQ, policy, or related content). 
    Your task is to extract structured metadata from this text using the following categories:

    1. **Document Type:** Identify if the content is an "FAQ", "Policy", "Terms & Conditions", "Guideline", or another type. If uncertain, use "Other".
    2. **Primary Topic/Subject:** Determine the broad subject or main theme. Select from the following categories: 
        - Data and Privacy
        - Home Page
        - Personal Loan
        - Scoins / Rewards
        - Transfer Money / UPI Transfer
        - Onboarding
        - Login
        - Learn
        - Miscellaneous
        - FD's / Investments
        - UPI
        - Deals
        - Referral
        - Errors
        - Credit Card / Card
    3. **Subtopic (Optional):** Provide a more granular categorization within the primary topic if applicable (e.g., for "Onboarding", it might be "Account Setup" or "Profile Verification").
    4. **Financial Product or Service:** Specify which product or service the content relates to. Select from the following categories:
        - Personal Loan
        - Scoins / Rewards
        - UPI
        - FD's / Investments
        - Credit Cards
        - Others (if not listed)
    5. **Keywords/Tags:** List key terms or tags summarizing the content (e.g., ["security", "fraud prevention", "customer support"]).

    Return only a valid JSON object in the following format:

    ```json
    {{
        "metadata": {{
            "document_type": "",
            "primary_topic": "",
            "subtopic": "",
            "financial_product": "",
            "keywords": []
        }}
    }}
    ```

    If a field is not applicable, return `null` instead of an empty string.

    text: {text}"""
)


def get_metadata_tags(rowdata: dict) -> dict:
    metadata_chain = prompt_template | model | JsonOutputParser()
    row_data_str = "\n".join([f"{key}: {str(rowdata[key])}" for key in rowdata.keys()])

    try:
        response = metadata_chain.invoke({"text": row_data_str})
        return response.get("metadata", {"metadata": None})
    except Exception as e:
        logging.error(f"Error processing row: {e}")
        return {"metadata": None}


def main():
    data_dir = os.path.join(os.getcwd(), "data_int")
    
    tsv_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(".tsv")]

    for tsv_file in tsv_files:
        try:
            df = pd.read_csv(tsv_file, sep="\t")
        except Exception as e:
            logging.error(f"Error reading file {tsv_file}: {e}")
            continue

        for index in df.index:
            rowdata = df.loc[index].to_dict()
            metadata = get_metadata_tags(rowdata)
            df.at[index, "metadata"] = str(metadata)

        output_file = tsv_file.replace(".tsv", "_with_metadata.csv")
        df.to_csv(output_file, index=False)

        logging.info(f"Successfully processed {tsv_file}, saved to {output_file}")


if __name__ == "__main__":
    main()
