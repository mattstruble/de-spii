from collections import defaultdict


class RedactionContext:
    def __init__(self, text: str) -> None:
        self.text: str = text
        self._pii_map: dict[str, str] = {}
        self._label_count: dict[str, int] = defaultdict(int)

    def _create_placeholder(self, label: str) -> str:
        """Create a unique placeholder for a PII value.

        Increments the internal counter to ensure placeholder uniqueness.

        :param label: The type of PII (e.g., 'EMAIL', 'PHONE_US', 'SSN')
        :return: A unique placeholder string in the format '<PII_{label}_{count}>'
        """
        self._label_count[label] += 1
        return f"<PII_{label}_{self._label_count[label]}>"

    def redact(self, pii: str, label: str) -> None:
        """Redacts the provided pii with the associated label.

        :param pii: PII string to be redacted.
        :param label: The type of PII (e.g., 'EMAIL', 'PHONE', 'SSN')
        """
        if pii not in self._pii_map.values():
            label = label.upper()
            placeholder = self._create_placeholder(label)
            self._pii_map[placeholder] = pii
            self.text = self.text.replace(pii, placeholder)

    def unredact(self, text: str) -> str:
        """Replace placeholders in the provided text with the original PII values.

        :param text: String containing placeholders.
        :return: The text with placeholders replaced by original PII values.
        """
        for placeholder, original in self._pii_map.items():
            text = text.replace(placeholder, original)
        return text
