import statistics
from copy import deepcopy
from dataclasses import dataclass, field

from .base import AgentMetrics, LLMMetrics, RealtimeModelMetrics, STTMetrics, TTSMetrics, EOUMetrics


@dataclass
class UsageSummary:
    llm_prompt_tokens: int
    llm_prompt_cached_tokens: int
    llm_completion_tokens: int
    tts_characters_count: int
    tts_audio_duration: float
    stt_audio_duration: float
    durations: dict = field(default_factory=lambda: {"llm": [], "tts": [], "stt": [], "eou": []})


class UsageCollector:
    def __init__(self) -> None:
        self._summary = UsageSummary(0, 0, 0, 0, 0.0, 0.0)

    def __call__(self, metrics: AgentMetrics) -> None:
        self.collect(metrics)

    def collect(self, metrics: AgentMetrics) -> None:
        if isinstance(metrics, LLMMetrics):
            self._summary.llm_prompt_tokens += metrics.prompt_tokens
            self._summary.llm_prompt_cached_tokens += metrics.prompt_cached_tokens
            self._summary.llm_completion_tokens += metrics.completion_tokens
            self._summary.durations["llm"].append(metrics.duration)

        elif isinstance(metrics, RealtimeModelMetrics):
            self._summary.llm_prompt_tokens += metrics.input_tokens
            self._summary.llm_prompt_cached_tokens += metrics.input_token_details.cached_tokens
            self._summary.llm_completion_tokens += metrics.output_tokens
            self._summary.durations["llm"].append(metrics.duration)

        elif isinstance(metrics, TTSMetrics):
            self._summary.tts_characters_count += metrics.characters_count
            self._summary.tts_audio_duration += metrics.audio_duration
            self._summary.durations["tts"].append(metrics.duration)

        elif isinstance(metrics, STTMetrics):
            self._summary.stt_audio_duration += metrics.audio_duration
            self._summary.durations["stt"].append(metrics.duration)

        elif isinstance(metrics, EOUMetrics):
            self._summary.durations["eou"].append(metrics.end_of_utterance_delay)

    # def get_summary(self) -> UsageSummary:
    #     return deepcopy(self._summary)
    def get_summary(self) -> dict:
        summary = deepcopy(self._summary.__dict__)
        stats = {}
        for module, vals in self._summary.durations.items():
            if vals:
                stats[module] = {
                    "mean": statistics.mean(vals),
                    "variance": statistics.variance(vals) if len(vals) > 1 else 0.0,
                }
        summary["stats"] = stats
        return summary
