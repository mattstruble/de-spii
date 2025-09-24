from dataclasses import dataclass


@dataclass
class DeSPII:
    text: str
    pii_map: dict[str, str]
    count: int = 0

    def create_placeholder(self, label: str) -> str:
        """Create a unique placeholder for a PII value.

        Increments the internal counter to ensure placeholder uniqueness.

        :param label: The type of PII (e.g., 'EMAIL', 'PHONE_US', 'SSN')
        :return: A unique placeholder string in the format '<PII_{label}_{count}>'
        """
        self.count += 1
        return f"<PII_{label}_{self.count}>"


def reconstruct_text(text: str, despii: DeSPII) -> str:
    reconstructed = text
    for placeholder, original_value in despii.pii_map.items():
        reconstructed = reconstructed.replace(placeholder, original_value)
    return reconstructed
