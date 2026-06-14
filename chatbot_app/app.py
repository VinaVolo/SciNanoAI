"""Backward-compat shim. The real UI lives in ``scinanoai.chatbot.ui``."""

from scinanoai.chatbot.ui import main

if __name__ == "__main__":
    main()
