"""Structured, user-facing errors.

Anything raised as :class:`UserFacingError` is presented to the user as three
layers: a plain-language summary of WHAT HAPPENED, a concrete WHAT TO DO NOW
(the ``remedy``, which the GUI turns into live buttons where it can perform the
fix itself), and a collapsible technical ``detail`` (paths, log tails,
tracebacks) intended for troubleshooting, never as the headline.

``kind`` is a stable machine-readable id so the GUI can attach the right
action buttons (e.g. "Repair local segmentation" / "Use cloud instead") and so
archived failure manifests can be grouped without parsing prose.
"""
from __future__ import annotations


class UserFacingError(RuntimeError):
    """An error with a human explanation and a prescribed next step."""

    # Known kinds (the GUI maps these to action buttons):
    #   local-runtime-broken  local seg runtime failed its health check
    #   seg-crashed           TotalSegmentator started but failed mid-run
    #   seg-oom               local segmentation likely ran out of memory
    #   cloud-unreachable     network / timeout talking to the cloud service
    #   cloud-rejected        the cloud service refused the request (4xx)
    #   no-series             no usable CT series in the chosen folder
    #   setup-failed          local runtime install / repair failed
    #   unknown               anything unanticipated

    def __init__(self, title: str, message: str, *, remedy: str = "",
                 detail: str = "", kind: str = "unknown"):
        super().__init__(message)
        self.title = title
        self.message = message
        self.remedy = remedy
        self.detail = detail
        self.kind = kind

    def __str__(self) -> str:  # readable in logs / CLI / non-GUI contexts
        parts = [self.message]
        if self.remedy:
            parts.append(f"What to do: {self.remedy}")
        if self.detail:
            parts.append(f"Details: {self.detail}")
        return "\n".join(parts)

    def to_manifest(self) -> dict:
        """Serializable form for the archived failure manifest."""
        return {"kind": self.kind, "title": self.title, "message": self.message,
                "remedy": self.remedy, "detail": self.detail}
