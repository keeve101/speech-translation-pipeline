from textual.app import App, ComposeResult
from textual.widgets import Footer, Label, Static
from textual.containers import Horizontal, ScrollableContainer, Vertical
from rich.text import Text

TEXT = """I must not fear.
Fear is the mind-killer.
Fear is the little-death that brings total obliteration.
I will face my fear.
I will permit it to pass over me and through me.
And when it has gone past, I will turn the inner eye to see its path.
Where the fear has gone there will be nothing. Only I will remain.
"""

colors = [
    "bright_white",
    "grey62",
    "yellow1",
    "bright_yellow",
]

def temp(a):
    t = Text()
    for i in range(9):
        t.append(TEXT, colors[2*((i+a)%2)])
    t.append(TEXT[:-15], colors[2*((1+a)%2)])
    return t

class TerminalText(App):
    """A Textual app to manage stopwatches."""

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]
    DEFAULT_CSS = """
    .header {
        dock: top;
        width: 100%;
        background: $panel;
        color: $foreground;
        height: 1;
        text-align: center;
    }
    """


    def __init__(self):
        super().__init__()
        self.languages = ['en', 'hi']

        self.history = {
            'en': temp(0),
            'hi': temp(1),
        }
        self.last_transcribed_lang = 'en'
        self.unconfirmed_transcription = TEXT[-15:]
        self.unconfirmed_translation = TEXT[-15:]

    def build_container(self, lang):
        text = self.unconfirmed_transcription if self.last_transcribed_lang == lang else self.unconfirmed_translation
        color = 2*int(self.last_transcribed_lang == lang) + 1
        return Vertical(
            Static(lang, classes="header"),
            ScrollableContainer(
                Label(self.history[lang] + Text(text, style=colors[color]))
            )
        )

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Horizontal(
            self.build_container('en'),
            self.build_container('hi'),
        )
        yield Footer()

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.theme = (
            "textual-dark" if self.theme == "textual-light" else "textual-light"
        )


if __name__ == "__main__":
    app = TerminalText()
    app.run()
