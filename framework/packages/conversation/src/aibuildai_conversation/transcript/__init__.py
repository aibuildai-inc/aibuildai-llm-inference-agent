"""Record public facts, consume trajectory JSON, or render default Markdown."""
from typing import TYPE_CHECKING
from .._impl.transcript import Transcript
from .._impl.trajectory import Trajectory

if TYPE_CHECKING:
    from .._impl.markdown import render_markdown

__all__ = ("Transcript", "Trajectory", "render_markdown")


def __getattr__(name: str) -> object:
    if name != "render_markdown":
        raise AttributeError(name)
    # Loading the optional default renderer does not make it a Web dependency.
    from .._impl.markdown import render_markdown
    return render_markdown
