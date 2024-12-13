from dataclasses import dataclass, field

@dataclass
class MockSettings:
    """Mock settings class for testing LLMSettings"""
    _temperature: float = field(default=0.7)
    _max_tokens: int = field(default=2000)
    _timeout: int = field(default=30)
    stream_chunk_size: int = field(default=1000)
    json_repair: bool = field(default=True)
    max_buffer_size: int = field(default=10000)

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        self.validate_temperature(value)
        self._temperature = value

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int):
        self.validate_max_tokens(value)
        self._max_tokens = value

    @property
    def timeout(self) -> int:
        return self._timeout

    @timeout.setter
    def timeout(self, value: int):
        self.validate_timeout(value)
        self._timeout = value

    @staticmethod
    def validate_temperature(value: float):
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError("Temperature must be a float between 0 and 1")

    @staticmethod
    def validate_max_tokens(value: int):
        if not isinstance(value, int) or value < 1:
            raise ValueError("max_tokens must be a positive integer")

    @staticmethod
    def validate_timeout(value: int):
        if not isinstance(value, int) or value < 1:
            raise ValueError("timeout must be a positive integer") 