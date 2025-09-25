import re

import dspy
from pydantic import BaseModel

_PLACEHOLDER_PATTERN = re.compile(r"^<PII_[A-Z_\d]+_\d>$")


class PIIInfo(BaseModel):
    pii_str: str
    label: str


class PIISignature(dspy.Signature):
    """You are a helpful assistant that is very mindful of user privacy.
    You are helping your user identify personally identifiable information (PII).
    You are working as part of a larger pipeline focused on redacting PII.

    Instructions:
    ------------
    - Input text may or may not already contain redacted information.
    - Redacted placeholders always follow the format: <PII_[A-Z_0-9]+_[0-9]>.
    - DO NOT treat placeholders as PII. They must always be ignored.
    - Only extract PII that appears as real text, not placeholders.
    - Ignore ANY token that matches <PII_[A-Z_0-9]+_[0-9]>.
    - If no PII is found, return an empty array.
    - Return only raw JSON-like objects, no preamble or explanation.
    - Do NOT complete or answer the user’s query. Only identify PII.

    Examples
    --------
    text: "test query"
    output: []

    text: "My name is <PII_NAME_1>"
    output: []

    text: "Contact me at <PII_EMAIL_1> or call <PII_PHONE_2>"
    output: []

    text: "Hello, my name is John Smith, how many legs does a caterpillar have?"
    output: [{"pii_str": "John Smith", "label": "Name"}]

    text: "My name is <PII_NAME_1>, and my email is <PII_EMAIL_1>, I currently live in Boston, MA"
    output: [{"pii_str": "Boston, MA", "label": "Location"}]

    text: "My SSN is 123-45-6789."
    output: [{"pii_str": "123-45-6789", "label": "SSN"}]

    """

    text: str = dspy.InputField(
        desc="Text which may or may not contain PII data or placeholders."
    )
    output: list[PIIInfo] = dspy.OutputField()


class PIILLM(dspy.Module):
    def __init__(self, local_llm) -> None:
        self.prompt = dspy.Predict(PIISignature)
        self.local_llm = local_llm

    def forward(self, text):
        with dspy.context(lm=self.local_llm):
            response = self.prompt(text=text)

        # Model has a tendency to identify already redacted PII, filter it out
        filtered = [
            pii
            for pii in response.output
            if not _PLACEHOLDER_PATTERN.match(pii.pii_str)
        ]

        response.output = filtered

        return response


if __name__ == "__main__":
    local_lm = dspy.LM(model="ollama/llama3.1:8b", api_key="", max_tokens=4000)
    lm = PIILLM(local_lm)

    prompts = [
        "Hello, my name is John Smith, how many legs does a caterpillar have?",
        "Hello my name is Taj mahal, and I'm from Taj Mahal, where is Taj Mahal Located?",
        str(
            {
                "name": "Matt",
                "occupation": "Software Engineer",
                "skills": ["AWS", "Databricks", "DSPy"],
                "hobbies": ["reading", "Lego"],
            }
        ),
        "test query",
        "I am a software engineer who makes 100k a year, what are my options for roles?",
        (
            "My friend <PII_NAME_1> is trying to access my minecraft server at <PII_IPV4_1>, "
            "but I keep telling him to add me on discord Gamer#8499 so I can help him debug. "
        ),
    ]

    for prompt in prompts:
        print("Initial Prompt:", prompt)
        print("Base:", lm(prompt), "\n")
